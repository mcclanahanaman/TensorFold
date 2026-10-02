"""Prompt hyper-connection read-outs on Jetson Thor (THOR-HC-FUSE): thor_hc's fused kernels against TensorFold's
read-out and an fp64 read-out, on real prompts.

    python tools/thor/hc_check.py kernels MODEL_DIR OUT_DIR [--prompts nat32k,syn128k] [--sizes 128,512,2048,4096]
        [--sample-rows 64]

The engine is built as served (int8 KV, --parallel 4, the Thor switches as they are in this process). On chosen
prompt passes (the first and the last of each prompt, and the middle one of a prompt over 16 passes) and layers (the
first, a third, two thirds and the last), each hyper-connection call's streams are kept before the served call; after
it, on the same inputs: the write-back (checked against the served streams), then for the first M rows, each M in
--sizes up to the pass's rows,

- TensorFold's prompt read-out (``forward._readout_b16``: glue.hc_normed, bf16.matmul's split-K sums, glue.hc_act,
  the up projection as the dense path runs it, glue.hc_mix) and thor_hc's (``down_act``, then ``upmix`` where the
  engine's startup check enabled it, else the up projection and ``mix``): the bytes of act, the inject gates and mixed
  compared (``equal_*``; both of thor_hc's mixes are also compared, ``equal_mixed_upmix`` and ``equal_mixed_mixh``,
  and the fused kernel's rounded up projection with the dense path's, ``equal_up_fused``);
- their per-row relative errors against an fp64 read-out of the same streams (rinv, the down projection, SiLU, the
  gates, the up projection and the mix in float64) on --sample-rows rows spread over the M: mean (``err_*``) and
  maximum (``errmax_*``);
- at M = the pass's rows, CUDA-event medians of 5 (``ms_*``): the write-back, each of TensorFold's kernels and its
  read-out as a whole (``ms_r25``), each of thor_hc's and its read-out as a whole (``ms_new``).

Writes OUT_DIR/hc_check.json: {"upmix": the engine's choice, "note": thor_hc's startup note, "rows": [...]};
tools/thor/hc_check_summary.py tabulates it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import ngram_residency as nr  # noqa: E402
import prefill_profile as pp  # noqa: E402

PROMPTS = {"syn32k": (32768, False, 4242 + 32768), "nat32k": (32768, True, 515151),
           "syn128k": (131072, False, 4242 + 131072), "nat128k": (131072, True, 525252)}


def timed(fn, reps: int = 5) -> float:
    fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b))
    return statistics.median(out)


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))


def rel(x: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """Per-row relative error (fp64 norms): mean and max over the rows."""

    e = (x.double() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-300)
    return float(e.mean()), float(e.max())


def ref64(hw: torch.Tensor, scale: torch.Tensor, down_w: torch.Tensor, up_w: torch.Tensor, streams: int, low: int,
          eps: float, inject: bool):
    """The read-out in float64 from the written-back streams hw [M, S*D] (bf16): act, inject gates, mixed."""

    m, wide = hw.shape
    d = wide // streams
    x = hw.double().view(m, streams, d)
    rinv = torch.rsqrt((x * x).mean(-1) + eps)
    normed = x * rinv[..., None] * scale.double().view(streams, d)
    v = (normed.reshape(m, wide) @ down_w.double().t()) / streams
    act = v[:, :low] * torch.sigmoid(v[:, :low])
    inj = 2.0 * torch.sigmoid(v[:, low:low + streams]) if inject else None
    up = act @ up_w.double().t()
    mixed = (torch.sigmoid(up).view(m, streams, d) * normed).mean(1)
    return act, inj, mixed


class Bench:
    """Wraps forward.hc_block: after the served call on a chosen pass and layer, both read-outs on the call's
    inputs (the streams kept before the call; the branch, MoE slots and inject gates are still intact then)."""

    def __init__(self, forward, glue, bf16, thor_hc, w, sizes: list[int], sample_rows: int) -> None:
        self.glue, self.bf16, self.th = glue, bf16, thor_hc
        self.sizes, self.sample_rows = sizes, sample_rows
        self.where = {}
        for layer in w.layers:
            for which in ("attn", "mlp"):
                hc = getattr(layer, f"{which}_hc", None)
                if hc is not None:
                    self.where[id(hc)] = (int(layer.index), which)
        n = len(w.layers)
        self.layers = sorted({int(w.layers[i].index) for i in (0, n // 3, (2 * n) // 3, n - 1)})
        self.first = id(w.layers[0].attn_hc)
        self.prompt, self.passes, self.passno = "", set(), -1
        self.results: list = []
        orig = forward.hc_block

        def hc_block(hc, b, R, eps, streams, low, mode, inject_prev, inject_out, h, branch=None, y=None, wts=None):
            if self.prompt and b.prefill and id(hc) == self.first:
                self.passno += 1
            pick = (self.prompt and b.prefill and self.passno in self.passes and id(hc) in self.where
                    and self.where[id(hc)][0] in self.layers)
            h0 = h[:R].clone() if pick else None
            orig(hc, b, R, eps, streams, low, mode, inject_prev, inject_out, h, branch=branch, y=y, wts=wts)
            if pick:
                layer, which = self.where[id(hc)]
                base = {"prompt": self.prompt, "pass": self.passno, "layer": layer, "hc": which, "mode": int(mode),
                        "R": int(R)}
                try:
                    self.bench(base, hc, R, eps, streams, low, mode, h0, h, branch, inject_prev, y, wts)
                except Exception as exc:  # noqa: BLE001
                    self.results.append({**base, "M": int(R), "bench_error": repr(exc)[-600:]})
                    print(f"[hc] bench failed: {repr(exc)[-600:]}", flush=True)

        forward.hc_block = hc_block

    def bench(self, base, hc, R, eps, S, low, mode, h0, h, branch, inject_prev, y, wts) -> None:
        glue, bf16, th = self.glue, self.bf16, self.th
        f32, bf = torch.float32, torch.bfloat16
        dev = h0.device
        wide = h0.shape[1]
        d = wide // S
        inject = hc.inject
        hw = h0.clone()
        pss = torch.empty((R, d // 256, S), dtype=f32, device=dev)
        glue.hc_writeback(hw, hw, pss, S, mode, branch=branch, inject=inject_prev, y=y, wts=wts)
        served = same(hw, h[:R])
        used = low + S if inject else low
        down_w, up_w = hc.down.b.weight[:used], hc.up.weight
        normed, up0, up1 = torch.empty_like(hw), torch.empty_like(hw), torch.empty_like(hw)
        xs_n = torch.empty((R, wide // 32), dtype=f32, device=dev)
        dn = torch.empty((R, hc.down.n), dtype=f32, device=dev)
        act0, act1 = (torch.empty((R, low), dtype=bf, device=dev) for _ in range(2))
        xs_a = torch.empty((R, low // 32), dtype=f32, device=dev)
        inj0, inj1 = ((torch.empty((R, S), dtype=bf, device=dev) if inject else None) for _ in range(2))
        mixed0, mixed1, mixed2 = (torch.empty((R, d), dtype=bf, device=dev) for _ in range(3))
        xs_m = torch.empty((R, d // 32), dtype=f32, device=dev)
        for M in [m for m in self.sizes if m <= R]:
            hm, pm = hw[:M], pss[:M]
            i0, i1 = (inj0[:M] if inject else None), (inj1[:M] if inject else None)

            def r25_normed():
                glue.hc_normed(hm, pm, hc.scale, normed[:M], xs_n[:M], S, eps)

            def r25_down():
                bf16.matmul(normed[:M], hc.down.b, out=dn[:M], f32=True)

            def r25_act():
                glue.hc_act(dn[:M], act0[:M], xs_a[:M], i0, S, low)

            def r25_up():
                th._up_ref(act0[:M], hc.up, up0[:M])

            def r25_mix():
                glue.hc_mix(up0[:M], normed[:M], mixed0[:M], xs_m[:M], S)

            def r25():
                for step in (r25_normed, r25_down, r25_act, r25_up, r25_mix):
                    step()

            def new_down():
                th.down_act(hm, pm, hc.scale, hc.down.b, act1[:M], i1, S, low, eps)

            def new_up():
                th._up_ref(act1[:M], hc.up, up1[:M])

            def new_mixh():
                th.mix(up1[:M], hm, pm, hc.scale, mixed2[:M], S, eps)

            def new_upmix():
                th.upmix(act1[:M], up_w, hm, pm, hc.scale, mixed1[:M], S, eps)

            def new():
                for step in ((new_down, new_upmix) if th.S.upmix else (new_down, new_up, new_mixh)):
                    step()

            for step in (r25, new_down, new_up, new_mixh, new_upmix):
                step()
            mixed_new = mixed1[:M] if th.S.upmix else mixed2[:M]
            row = {**base, "M": int(M), "equal_h_served": served,
                   "equal_act": same(act1[:M], act0[:M]),
                   "equal_inject": same(i1, i0) if inject else None,
                   "equal_mixed": same(mixed_new, mixed0[:M]),
                   "equal_mixed_upmix": same(mixed1[:M], mixed0[:M]),
                   "equal_mixed_mixh": same(mixed2[:M], mixed0[:M])}
            th.upmix(act1[:M], up_w, hm, pm, hc.scale, mixed2[:M], S, eps, up=up1[:M])   # the fused projection
            row["equal_up_fused"] = same(up1[:M], up0[:M])
            idx = torch.linspace(0, M - 1, min(self.sample_rows, M), device=dev).round().long().unique()
            a64, g64, m64 = ref64(hm[idx], hc.scale, down_w, up_w, S, low, eps, inject)
            for key, x0, x1, ref in (("act", act0[:M][idx], act1[:M][idx], a64),
                                     ("mixed", mixed0[:M][idx], mixed_new[idx], m64)):
                (row[f"err_{key}_r25"], row[f"errmax_{key}_r25"]) = rel(x0, ref)
                (row[f"err_{key}_new"], row[f"errmax_{key}_new"]) = rel(x1, ref)
            if inject:
                (row["err_inject_r25"], row["errmax_inject_r25"]) = rel(inj0[:M][idx], g64)
                (row["err_inject_new"], row["errmax_inject_new"]) = rel(inj1[:M][idx], g64)
            if M == R:
                hx, px = torch.empty_like(hw), torch.empty_like(pss)
                timings = {"wb": lambda: glue.hc_writeback(h0, hx, px, S, mode, branch=branch, inject=inject_prev,
                                                           y=y, wts=wts),
                           "r25_normed": r25_normed, "r25_down": r25_down, "r25_act": r25_act, "r25_up": r25_up,
                           "r25_mix": r25_mix, "r25": r25, "new_down_act": new_down, "new_up": new_up,
                           "new_mix": new_mixh, "new_upmix": new_upmix, "new": new}
                for key, fn in timings.items():
                    try:
                        row[f"ms_{key}"] = timed(fn)
                    except Exception as exc:  # noqa: BLE001
                        row[f"ms_{key}"] = repr(exc)[-200:]
            self.results.append(row)
            print("[hc] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                  flush=True)


def kernels(args) -> None:
    from tensorfold.families.qwen4_exp.cuda import bf16, forward, glue, thor_hc

    args.out.mkdir(parents=True, exist_ok=True)
    engine = pp.build_engine(args.model, 4, "int8")
    w = engine.w
    rows = int(getattr(engine.multi, "prefill_rows", 4096) or 4096)
    bench = Bench(forward, glue, bf16, thor_hc, w, [int(s) for s in args.sizes.split(",") if s], args.sample_rows)
    print(f"[hc] layers {bench.layers}; thor_hc ready {thor_hc.S.ready}, upmix {thor_hc.S.upmix} ({thor_hc.S.note}); "
          f"pass rows {rows}", flush=True)
    for name in [p for p in args.prompts.split(",") if p]:
        n, natural, seed = PROMPTS[name]
        passes = -(-n // rows)
        bench.passes = {0, passes - 1} | ({passes // 2} if passes > 16 else set())
        ids = pp.prompt_ids(args.model, n, seed, natural=natural)
        nr.forget(engine)
        bench.prompt, bench.passno = name, -1
        got = pp.run_prompt(engine, ids)
        bench.prompt = ""
        print(f"[hc] {name}: done, passes {sorted(bench.passes)} of {passes} ({got['tok_s']:.0f} tok/s with the "
              f"benches inside)", flush=True)
    out = {"upmix": thor_hc.S.upmix, "note": thor_hc.S.note, "layers": bench.layers, "rows": bench.results}
    (args.out / "hc_check.json").write_text(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("kernels")
    k.add_argument("model", type=Path)
    k.add_argument("out", type=Path)
    k.add_argument("--prompts", default="nat32k,syn128k")
    k.add_argument("--sizes", default="128,512,2048,4096")
    k.add_argument("--sample-rows", type=int, default=64)
    args = ap.parse_args()
    {"kernels": kernels}[args.cmd](args)


if __name__ == "__main__":
    main()
