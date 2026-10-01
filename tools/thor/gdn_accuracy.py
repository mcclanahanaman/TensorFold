"""G-A for Jetson Thor's DeltaNet prompt chain (THOR-K4): FlashInfer's chunked kernel and TensorFold's chain on real
activations of a real checkpoint, each against an fp64 reference of the same rule.

    python tools/thor/gdn_accuracy.py MODEL_DIR OUT_DIR [--tokens 17000] [--layers 0,9,18,27,35]
        [--drift 131072] [--drift-layers 0,18,35|all] [--natural-drift 65536] [--limit ROWS] [--survey]

The engine is built as served (int8 KV, --parallel 4); thor_gdn is turned off for the measured prompts, so every
DeltaNet layer runs TensorFold's chain and the chain inputs are TensorFold's own (q, k fp32; v bf16; g, beta fp32).
For the chosen DeltaNet layers (indices among the 36 DeltaNet layers), each pass's inputs, incoming state and
projection row are caught as the prompt fills (passes of 2048 rows, the state carried between them), and three
chains run on them, each carrying its own state from the prompt's first row:
  ref   fp64, one step a row (``gdn_reference.chain64``)
  tf    TensorFold's chain (fp32), whose state is checked against the engine's own at every pass
  thor  ``thor_gdn.chain`` on q and k rounded to fp16 and v (exact in fp16) as the Thor front writes them
Cells (G-A figure: thor's relative error over tf's, against ref):
  served  per pass: the read-out y, the gated RMSNorm output (TensorFold's back on each y; ref: back in fp64), the
          worst head's y, and the state at the pass end
  calls   the first prompt's rows re-run as calls of M = 128, 512, 2048, 4096 rows (and 1000, off the 64-row grid),
          the state carried; and as prompts of 63, 64, 65, 128, 512, 2047, 2048, 2049, 3000 and 4096 rows (passes of
          2048): outputs and last state; ``bits_as_served`` says whether thor's bits equal the served partition's
  decode  each pass's last state read out by the pass's last 8 queries (a decode step's read-out from there): in fp64,
          and rounded to bf16 as decode emits it; the prompt's last pass is the first decode step's
  drift   (--drift N) a second prompt of N tokens, per-pass errors over the whole prompt for --drift-layers
``--limit ROWS`` runs thor as thor_gdn serves it with that limit: the chunked kernel for a prompt's rows below ROWS,
TensorFold's chain from there on (thor_gdn.LIMIT is the served value); ``--survey`` skips the first prompt and its
cells (the drift prompt only, e.g. with --drift-layers all, to choose the limit).
Writes gdn_accuracy.json and ledger_metrics.json (err_ratio: the worst served y ratio; tests_passed/failed: cells).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import gdn_reference as ref  # noqa: E402
import ngram_residency as nr  # noqa: E402
import prefill_profile as pp  # noqa: E402

LIMIT = 1.5                                 # G-A: thor's error at most this times tf's
PROMPT_LENGTHS = (63, 64, 65, 128, 512, 2047, 2048, 2049, 3000, 4096)
CALLS = (128, 512, 2048, 4096, 1000)


def back64(y: torch.Tensor, p: torch.Tensor, norm: torch.Tensor, eps: float, nk: int, nv: int) -> torch.Tensor:
    """TensorFold's gated RMSNorm (gdn_io.cu back) in fp64 without its roundings: [n, nv * 128]."""

    conv = 2 * nk * 128 + nv * 128
    z = p[:, conv:conv + nv * 128].double().view(-1, nv, 128)
    yy = y.double()
    rinv = torch.rsqrt((yy * yy).mean(-1, keepdim=True) + eps)
    return (yy * rinv * norm.double() * torch.sigmoid(z)).reshape(y.shape[0], -1)


def head_worst(y: torch.Tensor, r: torch.Tensor) -> float:
    """The worst value head's relative error."""

    d = (y.double() - r).pow(2).sum(dim=(0, 2)).sqrt()
    return float((d / r.pow(2).sum(dim=(0, 2)).sqrt().clamp_min(1e-300)).max())


class Track:
    """One DeltaNet layer's three chains, each with its own state, and its caught inputs (first prompt only)."""

    def __init__(self, li: int, layer: int) -> None:
        self.li, self.layer = li, layer
        self.s_ref = self.s_tf = self.s_th = None
        self.rows, self.passes, self.inputs, self.engine_match = 0, [], [], True

    def start(self, s0: torch.Tensor) -> None:
        self.s_ref, self.s_tf, self.s_th = s0.double().clone(), s0.clone(), s0.clone()
        self.rows, self.passes, self.engine_match = 0, [], True


def thor_rows(q, k, v, gt, beta, s, at: int, limit: int | None):
    """Thor's path over rows [at, at + n): the chunked kernel below ``limit`` (fp16 q, k, v), TensorFold's chain from
    there (fp32 q, k; bf16 v). Returns ([(first row, y), ...], last state)."""

    from tensorfold.cuda.kernels import gdn as tf_gdn
    from tensorfold.cuda.kernels import thor_gdn

    n = q.shape[0]
    cut = n if not limit else min(n, max(0, limit - at))
    parts = []
    if cut > 0:
        f = torch.empty_like(s)
        parts.append((0, thor_gdn.chain(q[:cut].half(), k[:cut].half(), v[:cut].half(), gt[:cut].contiguous(),
                                        beta[:cut].contiguous(), s, f)))
        s = f
    if cut < n:
        f = torch.empty_like(s)
        parts.append((cut, tf_gdn.chain(q[cut:].contiguous(), k[cut:].contiguous(), v[cut:].contiguous(),
                                        gt[cut:].contiguous(), beta[cut:].contiguous(), s, f)))
        s = f
    return parts, s


def run_chains(q, k, v, gt, beta, s_ref, s_tf, s_th, at: int = 0, limit: int | None = None):
    """(y, last state) of ref and tf, and thor's ([(first row, y), ...], last state), each from its own state."""

    from tensorfold.cuda.kernels import gdn as tf_gdn

    y_ref, f_ref = ref.chain64(q, k, v, gt, beta, s_ref)
    f_tf = torch.empty_like(s_tf)
    y_tf = tf_gdn.chain(q, k, v, gt, beta, s_tf, f_tf)
    return (y_ref, f_ref), (y_tf, f_tf), thor_rows(q, k, v, gt, beta, s_th, at, limit)


def calls_run(fn, x: dict, s0: torch.Tensor, cuts: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """``fn`` over rows split at ``cuts``, the state carried: (outputs of every row, last state)."""

    ys, s = [], s0
    edges = [0, *cuts, x["q"].shape[0]]
    for a, b in zip(edges, edges[1:]):
        if b <= a:
            continue
        y, s = fn({key: t[a:b].contiguous() for key, t in x.items()}, s)
        ys.append(y)
    return torch.cat(ys), s


def tf_fn(x, s):
    from tensorfold.cuda.kernels import gdn as tf_gdn

    f = torch.empty_like(s)
    return tf_gdn.chain(x["q"], x["k"], x["v"], x["g"], x["beta"], s, f), f


def th_fn(x, s):
    from tensorfold.cuda.kernels import thor_gdn

    f = torch.empty_like(s)
    return thor_gdn.chain(x["q"].half(), x["k"].half(), x["v"].half(), x["g"], x["beta"], s, f), f


def readouts(q: torch.Tensor, f_ref, f_tf, f_th) -> dict:
    """The last 8 queries read out from each path's state: fp64 and bf16 (as decode emits) errors against ref's."""

    qn = q[-8:]
    r_ref = ref.readout64(f_ref, qn)
    out = {}
    for name, s in (("tf", f_tf), ("thor", f_th)):
        r = ref.readout64(s, qn)
        out[f"readout_{name}"] = ref.rel(r, r_ref)
        out[f"readout_bf16_{name}"] = ref.rel(r.bfloat16(), r_ref)
    return out


def ratio(a: float, b: float) -> float | None:
    return a / b if b else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--tokens", type=int, default=17000)
    ap.add_argument("--layers", default="0,9,18,27,35")
    ap.add_argument("--drift", type=int, default=0)
    ap.add_argument("--drift-layers", default="0,18,35")
    ap.add_argument("--limit", type=int, default=None, help="thor's chunked rows end here (default: thor_gdn.LIMIT)")
    ap.add_argument("--survey", action="store_true", help="the drift prompt only")
    ap.add_argument("--natural-drift", type=int, default=0, help="also a natural-text drift prompt of this many tokens")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    engine = pp.build_engine(args.model, 4, "int8")
    from tensorfold.cuda.kernels import thor_gdn
    from tensorfold.families.qwen4_exp.cuda import forward, gdn_io

    if not thor_gdn.S.ready:
        raise SystemExit("thor_gdn is not prepared in this process (not sm_110, or TENSORFOLD_THOR_GDN=0)")
    thor_gdn.set_active(False)                       # the engine runs TensorFold's chain: its inputs are its own
    limit = args.limit if args.limit is not None else getattr(thor_gdn, "LIMIT", None)
    w = engine.w
    cfg = w.cfg
    lin_layers = [layer.index for layer in w.layers if layer.linear]
    report: dict = {"model": str(args.model), "dtype": str(thor_gdn.S.dtype), "lin_layers": len(lin_layers),
                    "limit": limit, "prompts": []}
    orig_chain, orig_front = forward._prefill_chain, gdn_io.front
    ctx: dict = {"tracks": {}, "store": False}

    def hooked(g, st, li, b, a0, a1, c, cut=None):
        tr = ctx["tracks"].get(li)
        if tr is None:
            return orig_chain(g, st, li, b, a0, a1, c, cut)
        seen: dict = {}

        def front(*a, **kw):
            seen["x"] = orig_front(*a, **kw)
            return seen["x"]

        s_in = st.rec[st.cur[li], li].clone()
        p = b.proj[0, a0:a1].clone()
        gdn_io.front = front
        try:
            orig_chain(g, st, li, b, a0, a1, c, cut)
        finally:
            gdn_io.front = orig_front
        q, k, v, gt, beta = seen["x"]
        if tr.rows == 0:
            if bool(s_in.abs().max() != 0):
                raise RuntimeError("a prompt's first pass should start from a zero state")
            tr.start(s_in)
        else:
            tr.engine_match = tr.engine_match and bool(torch.equal(s_in, tr.s_tf))
        (y_ref, f_ref), (y_tf, f_tf), (parts, f_th) = run_chains(q, k, v, gt, beta, tr.s_ref, tr.s_tf, tr.s_th,
                                                                  tr.rows, limit)
        n, nv = q.shape[0], v.shape[1]
        y_th = torch.cat([y.float() for _, y in parts])
        outs = []
        for pieces in ([(0, y_tf)], parts):
            o = torch.empty((n, nv * cfg.dv), dtype=torch.bfloat16, device=p.device)
            xs = torch.empty((n, nv * cfg.dv // 32), dtype=torch.float32, device=p.device)
            for i, (a, y) in enumerate(pieces):
                b_ = pieces[i + 1][0] if i + 1 < len(pieces) else n
                gdn_io.back(y, p[a:b_], g.norm, c.eps, o[a:b_], xs[a:b_])
            outs.append(o)
        o_ref = back64(y_ref, p, g.norm, c.eps, cfg.nk, nv)
        row = {"rows": n, "at": tr.rows,
               "y_tf": ref.rel(y_tf, y_ref), "y_thor": ref.rel(y_th, y_ref),
               "head_tf": head_worst(y_tf, y_ref), "head_thor": head_worst(y_th, y_ref),
               "back_tf": ref.rel(outs[0], o_ref), "back_thor": ref.rel(outs[1], o_ref),
               "state_tf": ref.rel(f_tf, f_ref), "state_thor": ref.rel(f_th, f_ref),
               "chunked_rows": parts[1][0] if len(parts) > 1 else (n if parts[0][1].dtype == torch.float16 else 0),
               **readouts(q, f_ref, f_tf, f_th)}
        for key in ("y", "head", "back", "state", "readout", "readout_bf16"):
            row[key + "_x"] = ratio(row[key + "_thor"], row[key + "_tf"])
        tr.passes.append(row)
        if ctx["store"]:
            tr.inputs.append({"q": q, "k": k, "v": v, "g": gt, "beta": beta})
        tr.s_ref, tr.s_tf, tr.s_th, tr.rows = f_ref, f_tf, f_th, tr.rows + n

    forward._prefill_chain = hooked

    def prompt(n: int, seed: int, layers: list[int], store: bool, natural: bool = False) -> dict:
        ctx["tracks"] = {li: Track(li, lin_layers[li]) for li in layers}
        ctx["store"] = store
        nr.forget(engine)                            # nothing kept: the prompt fills from its first row
        ids = pp.prompt_ids(args.model, n, seed, natural=natural)
        got = pp.run_prompt(engine, ids)
        out = {"tokens": len(ids), "natural": natural, "seconds": got["seconds"], "layers": {}}
        for li, tr in ctx["tracks"].items():
            last = tr.passes[-1]
            dec = {key: last[key] for key in ("readout_tf", "readout_thor", "readout_bf16_tf", "readout_bf16_thor",
                                              "readout_x", "readout_bf16_x")}
            worst = {key: max(r[key] for r in tr.passes if r[key] is not None)
                     for key in ("y_x", "head_x", "back_x", "state_x", "readout_bf16_x")}
            out["layers"][str(li)] = {"layer": tr.layer, "passes": tr.passes, "decode": dec, "worst": worst,
                                      "engine_state_match": tr.engine_match}
            over = [r["at"] + r["rows"] for r in tr.passes if r["readout_bf16_x"] > 1.25]
            out["layers"][str(li)]["readout_over_1_25_from"] = over[0] if over else None
            print(f"[ga] {len(ids)} tokens, DeltaNet {li} (layer {tr.layer}): {len(tr.passes)} passes; worst y "
                  f"x{worst['y_x']:.3f} head x{worst['head_x']:.3f} back x{worst['back_x']:.3f} state "
                  f"x{worst['state_x']:.0f} read-out bf16 x{worst['readout_bf16_x']:.3f}; last pass y tf "
                  f"{last['y_tf']:.2e} thor {last['y_thor']:.2e}; decode bf16 x{dec['readout_bf16_x']:.3f} (fp64 x"
                  f"{dec['readout_x']:.0f}); read-out over 1.25x from row {over[0] if over else None}; engine "
                  f"state == tf {tr.engine_match}", flush=True)
        return out

    def pick(text: str) -> list[int]:
        return list(range(len(lin_layers))) if text == "all" else [int(x) for x in text.split(",")]

    cells: dict = {}
    if not args.survey:
        first = prompt(args.tokens, 4711, pick(args.layers), store=True)
        report["prompts"].append(first)
        (args.out / "gdn_accuracy.json").write_text(json.dumps(report, indent=1))

    # calls and prompt lengths on the caught rows
    for li, tr in (ctx["tracks"].items() if not args.survey else ()):
        x = {key: torch.cat([d[key] for d in tr.inputs]) for key in ("q", "k", "v", "g", "beta")}
        rows = x["q"].shape[0]
        s0 = torch.zeros_like(tr.s_tf)
        y_ref, f_ref = ref.chain64(x["q"], x["k"], x["v"], x["g"], x["beta"], s0)
        served = [r for r in range(2048, rows, 2048)]
        y_served, s_served = calls_run(th_fn, x, s0, served)
        layer_cells: dict = {}
        for m in CALLS:
            cuts = list(range(m, rows, m))
            y_tf, s_tf = calls_run(tf_fn, x, s0, cuts)
            y_th, s_th = calls_run(th_fn, x, s0, cuts)
            cell = {"y_tf": ref.rel(y_tf, y_ref), "y_thor": ref.rel(y_th, y_ref),
                    "state_tf": ref.rel(s_tf, f_ref), "state_thor": ref.rel(s_th, f_ref),
                    "bits_as_served": bool(torch.equal(y_th, y_served) and torch.equal(s_th, s_served))}
            cell["y_x"], cell["state_x"] = ratio(cell["y_thor"], cell["y_tf"]), ratio(cell["state_thor"],
                                                                                     cell["state_tf"])
            layer_cells[f"calls_{m}"] = cell
        for n in [n for n in PROMPT_LENGTHS if n <= rows]:
            xp = {key: t[:n].contiguous() for key, t in x.items()}
            yr, fr = ref.chain64(xp["q"], xp["k"], xp["v"], xp["g"], xp["beta"], s0)
            cuts = list(range(2048, n, 2048))
            y_tf, s_tf = calls_run(tf_fn, xp, s0, cuts)
            y_th, s_th = calls_run(th_fn, xp, s0, cuts)
            cell = {"y_tf": ref.rel(y_tf, yr), "y_thor": ref.rel(y_th, yr), "state_tf": ref.rel(s_tf, fr),
                    "state_thor": ref.rel(s_th, fr), "bits_as_served": bool(torch.equal(y_th, y_served[:n]))}
            cell["y_x"], cell["state_x"] = ratio(cell["y_thor"], cell["y_tf"]), ratio(cell["state_thor"],
                                                                                     cell["state_tf"])
            layer_cells[f"prompt_{n}"] = cell
        cells[str(li)] = layer_cells
        for name, cell in layer_cells.items():
            print(f"[ga] DeltaNet {li} {name}: y tf {cell['y_tf']:.2e} thor {cell['y_thor']:.2e} (x{cell['y_x']:.3f});"
                  f" state x{cell['state_x']:.0f}; bits as served {cell['bits_as_served']}", flush=True)
        tr.inputs = []
    report["cells"] = cells
    (args.out / "gdn_accuracy.json").write_text(json.dumps(report, indent=1))

    if args.drift:
        drift = prompt(args.drift, 8128, pick(args.drift_layers), store=False)
        for li, d in drift["layers"].items():
            ps = d["passes"]
            d["drift"] = {"y_x_first": ps[0]["y_x"], "y_x_last": ps[-1]["y_x"],
                          "y_thor_first": ps[0]["y_thor"], "y_thor_last": ps[-1]["y_thor"],
                          "state_thor_first": ps[0]["state_thor"], "state_thor_last": ps[-1]["state_thor"],
                          "state_thor_max": max(p["state_thor"] for p in ps)}
            print(f"[ga] drift {drift['tokens']} tokens, DeltaNet {li}: y thor {ps[0]['y_thor']:.2e} -> "
                  f"{ps[-1]['y_thor']:.2e} (x{ps[0]['y_x']:.3f} -> x{ps[-1]['y_x']:.3f}); state thor "
                  f"{ps[0]['state_thor']:.2e} -> {ps[-1]['state_thor']:.2e} (max {d['drift']['state_thor_max']:.2e})",
                  flush=True)
        report["prompts"].append(drift)
        (args.out / "gdn_accuracy.json").write_text(json.dumps(report, indent=1))
    if args.natural_drift:
        drift = prompt(args.natural_drift, 9137, pick(args.drift_layers), store=False, natural=True)
        report["prompts"].append(drift)
    forward._prefill_chain = orig_chain
    thor_gdn.set_active(True)

    # verdict: every served pass, every call/prompt cell, every decode read-out as emitted
    checks: dict = {}
    for pr in report["prompts"]:
        for li, d in pr["layers"].items():
            checks[f"{pr['tokens']} served y, DeltaNet {li}"] = d["worst"]["y_x"] <= LIMIT
            checks[f"{pr['tokens']} served back, DeltaNet {li}"] = d["worst"]["back_x"] <= LIMIT
            checks[f"{pr['tokens']} decode bf16 (every pass), DeltaNet {li}"] = d["worst"]["readout_bf16_x"] <= LIMIT
    for li, layer_cells in cells.items():
        for name, cell in layer_cells.items():
            checks[f"{name} y, DeltaNet {li}"] = cell["y_x"] is not None and cell["y_x"] <= LIMIT
    passed = sum(1 for v in checks.values() if v)
    report["checks"] = checks
    report["passed"], report["failed"] = passed, len(checks) - passed
    worst = max(d["worst"]["y_x"] for pr in report["prompts"] for d in pr["layers"].values())
    (args.out / "gdn_accuracy.json").write_text(json.dumps(report, indent=1))
    (args.out / "ledger_metrics.json").write_text(json.dumps({"err_ratio": worst, "tests_passed": passed,
                                                              "tests_failed": len(checks) - passed}))
    print(f"[ga] checks {passed}/{len(checks)} at <= {LIMIT}x (y, back, decode as emitted, call and prompt cells); "
          f"worst served y x{worst:.3f}; done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
