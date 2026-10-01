"""Flash Next's DeltaNet prompt chain on Jetson Thor: chunked candidates against TensorFold's serial chain (THOR-K4).

    python tools/thor/gdn_kernel_bench.py OUT.json [--rows 128,512,2048,4096] [--paths tf,fi_bf16,fi_f16,...]
        [--inputs CAPTURE.pt] [--props] [--io 2048]

Inputs are synthetic with Flash Next's shapes and statistics unless ``--inputs`` names a capture (a list of dicts with
q, k fp32 [T, 16, 128]; v bf16 [T, 48, 128]; g, beta fp32 [T, 48]; state fp32 [48, 128, 128], as
``prefill_profile.py --save-gdn`` writes them): 16 key heads (q, k: bf16 conv outputs L2-normed in fp32, q times
128^-0.5), 48 value heads (bf16), per-head decays g in (0, 1), beta = bf16(sigmoid), and a warmed-up initial state.

Paths (each times only its kernel, on inputs already in its dtypes):
  tf          ``tensorfold.cuda.kernels.gdn.chain``: one fp32 step a row (what prompts run today)
  fi_bf16     FlashInfer's chunked CuTe DSL kernel for SM100 (``chunk_gated_delta_rule_sm100``), bf16 q/k/v/y, fp32
  fi_f16      state and sums; the same with fp16 q/k/v/y. Its public entry dispatches on CC 9/10/12, not 11, so the
              kernel is called directly with a linear gate (it takes the log inside) and scale 1 (q comes scaled)
  fla_bf16    flash-linear-attention's Triton ``chunk_gated_delta_rule`` (fla-core, MIT), log-space gate, V-first state;
  fla_f16     fp32 = fp32 q/k/v with Triton's default fp32 dot precision
  fla_fp32

Each result is compared with an fp64 serial recurrence of the same rule on the same exact inputs: relative error of
the outputs (all rows and the last 64), of the last state, and of the first decode read-out from that state; the ratio
to ``tf`` is the G-A figure. Times are CUDA events (median of 7 rounds of 5 calls, each path after a second of GEMMs:
Thor's clocks drop while a kernel compiles). ``--props`` checks each chunked path's exactness properties on the
largest row count: the same call twice (determinism); one call against two calls split at 64-aligned rows with the
state handed over (aligned splits); the first m rows of a longer call against a call of m rows (partial chunks); and
the size of the change at an unaligned split. ``--io ROWS`` times Flash Next's DeltaNet front and back (gdn_io) on
ROWS synthetic rows: TensorFold's types (q, k fp32; v, y bf16) and Thor's (fp16), one and eight rows a block.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import gdn_reference as ref  # noqa: E402

HK, HV, D = 16, 48, 128


def synth(rows: int, seed: int) -> dict:
    """Flash Next-like chain inputs: q, k fp32 (bf16 conv outputs, L2-normed), v bf16, g and beta fp32."""

    gen = torch.Generator(device="cuda").manual_seed(seed)

    def bf(x):
        return x.bfloat16().float()

    def normed(x):
        return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)

    qa = bf(F.silu(torch.randn(rows, HK, D, generator=gen, device="cuda")))
    ka = bf(F.silu(torch.randn(rows, HK, D, generator=gen, device="cuda")))
    q = normed(qa) * (1.0 / math.sqrt(D))
    k = normed(ka)
    v = F.silu(1.5 * torch.randn(rows, HV, D, generator=gen, device="cuda")).bfloat16()
    scale = torch.logspace(-3, 0, HV, device="cuda")[torch.randperm(HV, generator=gen, device="cuda")]
    g = torch.exp(-scale * F.softplus(torch.randn(rows, HV, generator=gen, device="cuda")))
    beta = bf(torch.sigmoid(torch.randn(rows, HV, generator=gen, device="cuda")))
    return {"q": q.contiguous(), "k": k.contiguous(), "v": v.contiguous(), "g": g.contiguous(),
            "beta": beta.contiguous()}


def warm_state(seed: int) -> torch.Tensor:
    """A realistic initial state: the fp64 chain's last state after 1024 synthetic rows, as fp32."""

    c = synth(1024, seed)
    _, s = ref.chain64(c["q"], c["k"], c["v"], c["g"], c["beta"], torch.zeros(HV, D, D, device="cuda"))
    return s.float().contiguous()


# ---------------------------------------------------------------------------------------------------- the paths

class TF:
    name = "tf"

    def prep(self, c: dict) -> dict:
        return dict(c)

    def __call__(self, c: dict, s0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from tensorfold.cuda.kernels import gdn as tf_gdn

        final = torch.empty_like(s0)
        y = tf_gdn.chain(c["q"], c["k"], c["v"], c["g"], c["beta"], s0, final)
        return y, final


class FI:
    def __init__(self, dtype: torch.dtype) -> None:
        self.dtype = dtype
        self.name = "fi_bf16" if dtype == torch.bfloat16 else "fi_f16"

    def prep(self, c: dict) -> dict:
        t = c["q"].shape[0]
        return {"q": c["q"].to(self.dtype).contiguous(), "k": c["k"].to(self.dtype).contiguous(),
                "v": c["v"].to(self.dtype).contiguous(), "g": c["g"], "beta": c["beta"],
                "cu": torch.tensor([0, t], dtype=torch.int32, device="cuda")}

    def __call__(self, c: dict, s0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from flashinfer.gdn_kernels import chunk_gated_delta_rule_sm100 as fi100

        t = c["q"].shape[0]
        y = torch.empty((t, HV, D), dtype=self.dtype, device="cuda")
        st = torch.empty((1, HV, D, D), dtype=torch.float32, device="cuda")
        fi100(c["q"], c["k"], c["v"], c["g"], c["beta"], y, c["cu"], s0[None], st, 1.0)
        return y, st[0]


class FLA:
    def __init__(self, dtype: torch.dtype) -> None:
        self.dtype = dtype
        self.name = {torch.bfloat16: "fla_bf16", torch.float16: "fla_f16", torch.float32: "fla_fp32"}[dtype]

    def prep(self, c: dict) -> dict:
        return {"q": c["q"].to(self.dtype)[None].contiguous(), "k": c["k"].to(self.dtype)[None].contiguous(),
                "v": c["v"].to(self.dtype)[None].contiguous(), "g": torch.log(c["g"])[None].contiguous(),
                "beta": c["beta"][None].contiguous()}

    def __call__(self, c: dict, s0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        y, st = chunk_gated_delta_rule(c["q"], c["k"], c["v"], g=c["g"], beta=c["beta"], scale=1.0,
                                       initial_state=s0[None], output_final_state=True, state_v_first=True)
        return y[0], st[0]


def make(name: str):
    return {"tf": TF, "fi_bf16": lambda: FI(torch.bfloat16), "fi_f16": lambda: FI(torch.float16),
            "fla_bf16": lambda: FLA(torch.bfloat16), "fla_f16": lambda: FLA(torch.float16),
            "fla_fp32": lambda: FLA(torch.float32)}[name]()


def _batch_first(c: dict) -> bool:
    return c["q"].dim() == 4


def sub(c: dict, a: int, b: int) -> dict:
    if _batch_first(c):
        return {k: (x[:, a:b].contiguous() if k != "cu" else x) for k, x in c.items()}
    out = {k: (x[a:b].contiguous() if k != "cu" else torch.tensor([0, b - a], dtype=torch.int32, device="cuda"))
           for k, x in c.items()}
    return out


def spin(seconds: float = 1.0) -> None:
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    end = time.time() + seconds
    while time.time() < end:
        x @ x
    torch.cuda.synchronize()


def timed(fn, rounds: int = 7, iters: int = 5) -> float:
    fn()
    torch.cuda.synchronize()
    spin()
    out = []
    for _ in range(rounds):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / iters)
    return statistics.median(out)


def errors(y: torch.Tensor, s: torch.Tensor, ry: torch.Tensor, rs: torch.Tensor, qn: torch.Tensor) -> dict:
    tail = min(64, y.shape[0])
    return {"out_err": ref.rel(y, ry), "tail_err": ref.rel(y[-tail:], ry[-tail:]), "state_err": ref.rel(s, rs),
            "readout_err": ref.rel(ref.readout64(s, qn), ref.readout64(rs, qn))}


def properties(path, c: dict, s0: torch.Tensor) -> dict:
    """Bit-exactness properties of a chunked path on prepared inputs ``c`` of T rows."""

    t = (c["q"].shape[1] if _batch_first(c) else c["q"].shape[0])
    y, s = path(c, s0)
    y2, s2 = path(c, s0)
    out: dict = {"deterministic": bool(torch.equal(y, y2) and torch.equal(s, s2))}
    for p in [p for p in (64, 128, 1024, 1088, t - 64, t - 128) if 0 < p < t]:
        ya, sa = path(sub(c, 0, p), s0)
        yb, sb = path(sub(c, p, t), sa.contiguous())
        yy = torch.cat([ya, yb], dim=0)
        out[f"aligned_split_{p}"] = bool(torch.equal(yy, y) and torch.equal(sb, s))
        if not out[f"aligned_split_{p}"]:
            out[f"aligned_split_{p}_maxdiff"] = [float((yy.float() - y.float()).abs().max()),
                                                 float((sb - s).abs().max())]
    for m in [m for m in (1, 63, 65, 1000, t - 1) if 0 < m < t]:
        ym, _ = path(sub(c, 0, m), s0)
        same = bool(torch.equal(ym, y[:m]))
        out[f"partial_{m}"] = same
        if not same:
            out[f"partial_{m}_maxdiff"] = float((ym.float() - y[:m].float()).abs().max())
    for p in [p for p in (1000, 1001) if 0 < p < t]:
        ya, sa = path(sub(c, 0, p), s0)
        yb, sb = path(sub(c, p, t), sa.contiguous())
        yy = torch.cat([ya, yb], dim=0)
        out[f"unaligned_split_{p}_maxdiff"] = [float((yy.float() - y.float()).abs().max()),
                                               float((sb - s).abs().max())]
    return out


def io_bench(rows: int) -> dict:
    """Front and back times (ms) per type and rows a block, on synthetic projection rows."""

    from tensorfold.families.qwen4_exp.cuda import gdn, gdn_io

    gen = torch.Generator(device="cuda").manual_seed(5)
    p = (torch.randn((rows, gdn.PW), generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    cs = (torch.randn((3, gdn.CONV), generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    cw = (torch.randn((gdn.CONV, 4), generator=gen, device="cuda") * 0.3).to(torch.bfloat16)
    a_log = torch.randn((gdn.NV,), generator=gen, device="cuda") * 0.5
    dt = torch.randn((gdn.NV,), generator=gen, device="cuda") * 0.5
    nw = (1 + 0.1 * torch.randn((gdn.DV,), generator=gen, device="cuda")).to(torch.bfloat16)
    win = torch.tensor([[j + t if j + t < 3 else j + t for t in range(4)] for j in range(rows)], dtype=torch.int32,
                       device="cuda")
    sid = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ptrs = torch.tensor([cs.data_ptr()], dtype=torch.int64, device="cuda")
    out: dict = {}
    for name, qk, vt in (("tf", torch.float32, torch.bfloat16), ("thor", torch.float16, torch.float16)):
        bufs = (torch.empty((rows, gdn.NK, gdn.DK), dtype=qk, device="cuda"),
                torch.empty((rows, gdn.NK, gdn.DK), dtype=qk, device="cuda"),
                torch.empty((rows, gdn.NV, gdn.DV), dtype=vt, device="cuda"),
                torch.empty((rows, gdn.NV), device="cuda"), torch.empty((rows, gdn.NV), device="cuda"))
        y = (torch.randn((rows, gdn.NV, gdn.DV), generator=gen, device="cuda") * 0.3).to(vt)
        o = torch.empty((rows, gdn.NV * gdn.DV), dtype=torch.bfloat16, device="cuda")
        xs = torch.empty((rows, gdn.NV * gdn.DV // 32), device="cuda")
        for rb in (1, 8):
            out[f"front_{name}_rb{rb}"] = timed(lambda: gdn_io.front(p, ptrs, sid, win, cw, a_log, dt, gdn.NK,
                                                                     out=bufs, rows_per_block=rb))
            out[f"back_{name}_rb{rb}"] = timed(lambda: gdn_io.back(y, p, nw, 1e-6, o, xs, rows_per_block=rb))
    print(f"[gdn] io {rows} rows (ms): " + ", ".join(f"{k} {v:.3f}" for k, v in out.items()), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--rows", default="128,512,2048,4096")
    ap.add_argument("--paths", default="tf,fi_bf16,fi_f16,fla_bf16,fla_f16,fla_fp32")
    ap.add_argument("--inputs", type=Path, default=None)
    ap.add_argument("--props", action="store_true")
    ap.add_argument("--io", type=int, default=0, help="time the DeltaNet front and back on this many rows, then exit")
    args = ap.parse_args()
    spin(3.0)
    if args.io:
        args.out.write_text(json.dumps({"io": io_bench(args.io)}, indent=1))
        return
    report: dict = {"device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
                    "torch": torch.__version__, "rows": {}}
    for mod in ("flashinfer", "triton", "fla", "cutlass"):
        try:
            report[mod] = getattr(__import__(mod), "__version__", "?")
        except Exception as exc:                          # noqa: BLE001
            report[mod] = f"unavailable: {exc!r}"[:200]
    paths = []
    for name in args.paths.split(","):
        try:
            paths.append(make(name))
        except Exception as exc:                          # noqa: BLE001
            report.setdefault("skipped", {})[name] = repr(exc)[:300]
    caps = torch.load(str(args.inputs), map_location="cuda") if args.inputs else None
    s0 = caps[0]["state"].float().contiguous() if caps else warm_state(7)
    for rows in [int(r) for r in args.rows.split(",")]:
        if caps:
            full = {k: torch.cat([c[k] for c in caps]) for k in ("q", "k", "v", "g", "beta")}
            c = {k: x[:rows].contiguous() for k, x in full.items()}
        else:
            c = synth(rows, 100 + rows)
        ry, rs = ref.chain64(c["q"], c["k"], c["v"], c["g"], c["beta"], s0)
        qn = synth(8, 3)["q"]                            # next queries: the first decode read-outs
        row: dict = {}
        for path in paths:
            try:
                pc = path.prep(c)
                y, s = path(pc, s0)
                torch.cuda.synchronize()
                row[path.name] = {"ms": timed(lambda: path(pc, s0)), **errors(y, s, ry, rs, qn)}
            except Exception as exc:                      # noqa: BLE001
                row[path.name] = {"error": repr(exc)[:600]}
        base = row.get("tf", {})
        for r in row.values():
            if "ms" in r and "ms" in base:
                r["speedup"] = base["ms"] / r["ms"]
                for key in ("out_err", "tail_err", "state_err", "readout_err"):
                    r[key + "_x"] = r[key] / base[key] if base[key] else None
        report["rows"][str(rows)] = row
        for name, r in row.items():
            if "ms" in r:
                print(f"[gdn] rows {rows} {name}: {r['ms']:.3f} ms (x{r.get('speedup', 1):.2f}); out "
                      f"{r['out_err']:.2e} "
                      f"(x{r.get('out_err_x') or 1:.2f}) tail {r['tail_err']:.2e} (x{r.get('tail_err_x') or 1:.2f}) "
                      f"state {r['state_err']:.2e} (x{r.get('state_err_x') or 1:.1f}) readout {r['readout_err']:.2e} "
                      f"(x{r.get('readout_err_x') or 1:.2f})", flush=True)
            else:
                print(f"[gdn] rows {rows} {name}: {r['error']}", flush=True)
        args.out.write_text(json.dumps(report, indent=1))
    if args.props:
        rows = max(int(r) for r in args.rows.split(","))
        c = synth(rows, 100 + rows) if not caps else {k: torch.cat([x[k] for x in caps])[:rows].contiguous()
                                                       for k in ("q", "k", "v", "g", "beta")}
        report["props"] = {}
        for path in paths:
            if path.name == "tf":
                continue
            try:
                report["props"][path.name] = properties(path, path.prep(c), s0)
            except Exception as exc:                      # noqa: BLE001
                report["props"][path.name] = {"error": repr(exc)[:600]}
            print(f"[gdn] props {path.name} ({rows} rows): {report['props'][path.name]}", flush=True)
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
