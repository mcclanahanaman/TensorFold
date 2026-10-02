"""Routed-expert glue on Jetson Thor (THOR-EXPERT-GLUE): thor_glue's path (no scatter: the write-back reads the
experts' permuted rows; the shared expert's SwiGLU in one kernel) against the scatter path (r28's kernels) and fp64,
on real prompts.

    python tools/thor/glue_check.py kernels MODEL_DIR OUT_DIR [--prompts nat32k,syn128k] [--sizes 128,512,2048,4096]
        [--sample-rows 64]

The engine is built as served (int8 KV, --parallel 4, the Thor switches as they are in this process). On chosen
prompt passes (the first and the last of each prompt, and the middle one of a prompt over 16 passes) and layers (the
first, a third, two thirds and the last), after the served MoE call, on its inputs (the experts' input rows b.mixed,
the streams b.h and the inject gates b.inj_m, which the call leaves as they are), for the first M rows, each M in
--sizes up to the pass's rows, on MoE buffers of the check's own:

- the scatter path (``thor_glue.set_active(False)``: thor_experts' scatter into y, the shared expert's torch SwiGLU and
  copies, then ``glue.hc_writeback`` mode 2 on y) and thor_glue's (``set_active(True)``, the write-back on the kept
  rows): the bytes of the written-back streams and their square sums (``equal_h``, ``equal_pss``), of the shared
  expert's act and output (``equal_act``, ``equal_shared``); at M = the pass's rows also the served call's own kept rows
  written back (``equal_served``);
- per-row relative errors against fp64 on --sample-rows rows spread over the M: the written-back streams against the
  streams plus the fp64 weighted sum of the same slots times the gates (``err_h_off_*``, ``err_h_on_*``), the shared
  act against fp64 SiLU(gate) x up of the same projection (``err_act_off_*``, ``err_act_on_*``): mean and max;
- at M = the pass's rows, CUDA-event medians of 5 (``ms_*``): the MoE call (``moe``), the write-back (``wb``), the
  scatter kernel alone (``scatter``) and the shared expert from its gate/up projection on (``shared``), each path.

Writes OUT_DIR/glue_check.json: {"note": thor_glue's startup note, "active": whether the engine served with it,
"layers": [...], "rows": [...]}; tools/thor/glue_check_summary.py tabulates it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import ngram_residency as nr  # noqa: E402
import prefill_profile as pp  # noqa: E402
from hc_check import PROMPTS, rel, same, timed  # noqa: E402


class Bench:
    """Wraps forward.moe_block: after the served call on a chosen pass and layer, both paths on the call's inputs."""

    def __init__(self, w, sizes: list[int], sample_rows: int) -> None:
        from tensorfold.families.qwen4_exp.cuda import forward

        self.sizes, self.sample_rows = sizes, sample_rows
        n = len(w.layers)
        self.layers = sorted({int(w.layers[i].index) for i in (0, n // 3, (2 * n) // 3, n - 1)})
        self.index = {id(layer): int(layer.index) for layer in w.layers}
        self.first = id(w.layers[0])
        self.prompt, self.passes, self.passno = "", set(), -1
        self.results: list = []
        self.buf = None
        orig = forward.moe_block

        def moe_block(layer, w_, b, R):
            if self.prompt and b.prefill and id(layer) == self.first:
                self.passno += 1
            pick = (self.prompt and b.prefill and self.passno in self.passes
                    and self.index.get(id(layer)) in self.layers)
            out = orig(layer, w_, b, R)
            if pick:
                base = {"prompt": self.prompt, "pass": self.passno, "layer": self.index[id(layer)], "R": int(R)}
                try:
                    self.bench(base, layer, w_, b, R, out)
                except Exception as exc:  # noqa: BLE001
                    self.results.append({**base, "M": int(R), "bench_error": repr(exc)[-600:]})
                    print(f"[glue] bench failed: {repr(exc)[-600:]}", flush=True)
            return out

        forward.moe_block = moe_block

    def buffers(self, w, dev, rows: int):
        """One prompt MoE buffer set with the Thor experts' scratch, the check's own (the served ones stay intact)."""

        from tensorfold.cuda import moe as moe_mod
        from tensorfold.cuda.nvfp4 import thor_experts as thor
        from tensorfold.families.qwen4_exp.cuda.state import _MoECfg

        if self.buf is None or self.buf.rows < rows:
            cfg = _MoECfg(w.cfg)
            buf = moe_mod.MoEBuffers(rows, cfg, dev, prefill=True)
            top = int(cfg.num_experts_per_tok)
            ex = w.layers[0].moe.experts
            buf.thor = thor.Scratch(rows, top, ex.routed, ex.width, ex.dims, dev)
            self.buf, self.cfg = buf, cfg
        return self.buf

    def bench(self, base, layer, w, b, R: int, served) -> None:
        from tensorfold.cuda.nvfp4 import thor_experts as thor
        from tensorfold.families.qwen4_exp.cuda import glue, nvfp4_moe, thor_glue

        m = layer.moe
        ex = m.experts
        if not isinstance(ex.routed_experts, thor.ThorExperts):
            return
        f64 = torch.float64
        x, h, inj = b.mixed[:R], b.h[:R], b.inj_m[:R]
        dev = x.device
        S = h.shape[1] // ex.dims
        buf = self.buffers(w, dev, R)
        cfg, top = self.cfg, buf.slots - 1
        was = thor_glue.S.active
        sizes = [s for s in self.sizes if s <= R] or [R]
        if sizes[-1] != R and R not in self.sizes:
            sizes.append(R)
        try:
            for M in sizes:
                row = {**base, "M": int(M)}
                got, y_off = {}, None
                for on in (False, True):
                    thor_glue.set_active(on)
                    nvfp4_moe.moe(x[:M], None, m.router, ex, buf, cfg, unscattered=True)
                    if on != (buf.kept is not None):
                        raise RuntimeError(f"the path did not follow the switch (on={on}, kept={buf.kept is not None})")
                    slots = buf.kept if on else buf.y[:M]
                    hh = h[:M].clone()
                    pss = torch.full((M, ex.dims // 256, S), float("nan"), dtype=torch.float32, device=dev)
                    glue.hc_writeback(hh, hh, pss, S, 2, inject=inj[:M], y=slots, wts=buf.wts[:M])
                    shared = (buf.kept.shared if on else buf.y[:M, top]).clone()
                    got[on] = (hh, pss, buf.act[:M, top].clone(), shared)
                    if not on:
                        y_off, wts = buf.y[:M].clone(), buf.wts[:M].clone()
                    if on and M == R:          # the served call's own kept rows, written back the same way
                        if served[0] == 2 and not isinstance(served[1], torch.Tensor):
                            hs = h[:M].clone()
                            ps = torch.full_like(pss, float("nan"))
                            glue.hc_writeback(hs, hs, ps, S, 2, inject=inj[:M], y=served[1], wts=served[2])
                            row["equal_served"] = same(hs, hh) and same(ps, pss)
                        else:
                            row["equal_served"] = None
                    if M == R:
                        row.update(self.timings(on, x[:M], ex, m.router, buf, cfg, h[:M], inj[:M], S, top))
                (h0, p0, a0, s0), (h1, p1, a1, s1) = got[False], got[True]
                row.update({"equal_h": same(h0, h1), "equal_pss": same(p0, p1), "equal_act": same(a0, a1),
                            "equal_shared": same(s0, s1)})
                idx = torch.linspace(0, M - 1, min(self.sample_rows, M), device=dev).round().long().unique()
                branch = (y_off[idx].to(f64) * wts[idx].to(f64)[:, :, None]).sum(1)              # [n, D]
                want = (h[idx].to(f64).view(-1, S, ex.dims) + branch[:, None, :] * inj[idx].to(f64)[:, :, None])
                want = want.reshape(len(idx), -1)
                for tag, hv in (("off", h0), ("on", h1)):
                    row[f"err_h_{tag}_mean"], row[f"err_h_{tag}_max"] = rel(hv[idx], want)
                g = nvfp4_moe._shared(ex.shared.gu, x[:M][idx], True, False).to(f64)
                gate, up = g[:, :ex.width], g[:, ex.width:]
                act64 = gate * torch.sigmoid(gate) * up
                for tag, av in (("off", a0), ("on", a1)):
                    row[f"err_act_{tag}_mean"], row[f"err_act_{tag}_max"] = rel(av[idx], act64)
                self.results.append(row)
                print(f"[glue] {base['prompt']} pass {base['pass']} layer {base['layer']} M {M}: h {row['equal_h']} "
                      f"pss {row['equal_pss']} act {row['equal_act']} shared {row['equal_shared']}"
                      + (f" served {row.get('equal_served')}" if M == R else ""), flush=True)
        finally:
            thor_glue.set_active(was)

    @staticmethod
    def timings(on: bool, x, ex, router, buf, cfg, h, inj, S: int, top: int) -> dict:
        """CUDA-event medians on this path: the MoE call, the write-back, the scatter alone, the shared expert."""

        from tensorfold.cuda.nvfp4 import thor_experts as thor
        from tensorfold.families.qwen4_exp.cuda import glue, nvfp4_moe, thor_glue

        tag = "on" if on else "off"
        M = x.shape[0]
        hh = h.clone()
        pss = torch.empty((M, ex.dims // 256, S), dtype=torch.float32, device=x.device)
        out = {f"ms_moe_{tag}": timed(lambda: nvfp4_moe.moe(x, None, router, ex, buf, cfg, unscattered=True))}
        slots = buf.kept if on else buf.y[:M]
        out[f"ms_wb_{tag}"] = timed(lambda: glue.hc_writeback(hh, hh, pss, S, 2, inject=inj, y=slots,
                                                               wts=buf.wts[:M]))
        if on:
            ys = buf.y.view(-1, ex.dims)[:M]

            def shared():
                thor_glue.swiglu(nvfp4_moe._shared(ex.shared.gu, x, True, False), ex.width, buf.act[:M, top])
                nvfp4_moe._shared(ex.shared.down, buf.act[:M, top], True, False, out=ys)
        else:
            def shared():
                buf.act[:M, top] = ex.shared_act(x, True)
                buf.y[:M, top] = nvfp4_moe._shared(ex.shared.down, buf.act[:M, top], True, False)

            from tensorfold.cuda import experts as grouped

            routed, plan, scr = ex.routed_experts, buf.plan, buf.thor
            tile, _ = thor.tactic(M)
            grid = (grouped.max_items(M * buf.slots, routed.count + 1, tile),)
            _, _, scatter = thor._glue()
            out["ms_scatter"] = timed(lambda: scatter[grid](scr.hidden, plan.members, plan.items, scr.tiles,
                                                            buf.y.view(-1, routed.dims), T=tile, D=routed.dims,
                                                            BD=thor._COPY, num_warps=4))
        out[f"ms_shared_{tag}"] = timed(shared)
        return out


def kernels(args) -> None:
    from tensorfold.families.qwen4_exp.cuda import thor_glue

    args.out.mkdir(parents=True, exist_ok=True)
    engine = pp.build_engine(args.model, 4, "int8")
    w = engine.w
    rows = int(getattr(engine.multi, "prefill_rows", 4096) or 4096)
    bench = Bench(w, [int(s) for s in args.sizes.split(",") if s], args.sample_rows)
    active = thor_glue.S.active
    print(f"[glue] layers {bench.layers}; thor_glue ready {thor_glue.S.ready}, active {active} ({thor_glue.S.note}); "
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
        print(f"[glue] {name}: done, passes {sorted(bench.passes)} of {passes} ({got['tok_s']:.0f} tok/s with the "
              f"benches inside)", flush=True)
    out = {"note": thor_glue.S.note, "active": active, "layers": bench.layers, "rows": bench.results}
    (args.out / "glue_check.json").write_text(json.dumps(out, indent=1))


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
