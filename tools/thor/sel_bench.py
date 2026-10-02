"""THOR-K4-QSA-SEL tuning bench: real block-score slices from served prompt passes, then every selection kernel on them.

    python tools/thor/sel_bench.py MODEL_DIR OUT_DIR [--prompts nat32k,syn128k] [--p0 2048,7936,16128,32512]
        [--layers 3] [--reps 10]

Captures, while the engine as served (int8 KV, --parallel 4, the Thor switches as set in this process) fills each
prompt, the block scores of the 256-row slices that start at each p0, for the first ``--layers`` attention layers
(sparse rows' selections are made from exactly these scores). Then, with the engine released, times on each slice
with CUDA events (median of ``--reps`` after one warm call): TensorFold's ``_launch_select``, the Triton selection
(``_select_rows``) and thor_select.cu built with THOR_SELECT_TUNING at every (threads, keys) config, with an 11- and a
13-bit first digit and the profiling stops (after step 0, after step 2), checking every full selection's lists, key
counts and sparse flags against TensorFold's. Also records how the first digit spreads (distinct digits a row, the
largest multiplicity among 32 consecutive blocks: a warp's atomic conflicts). Writes sel_bench.json; prints a table.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).parent))

CONFIGS = {1: (128, 8), 2: (128, 16), 3: (256, 16), 4: (256, 32), 5: (512, 32), 6: (512, 64), 7: (1024, 32),
           8: (1024, 16), 9: (512, 16), 10: (256, 64), 11: (1024, 8)}


def timed(fn, reps: int) -> float:
    """The median of ``reps`` CUDA-event timings of ``fn`` (after one warm call), in ms."""

    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]


def tuning_ext():
    from tensorfold.cuda.build import load
    from tensorfold.families.qwen4_exp.cuda import thor_qsa

    here = Path(thor_qsa.__file__).parent
    return load(name="tensorfold_qwen4_exp_thor_select_tuning",
                sources=[str(here / "thor_select.cpp"), str(here / "thor_select.cu")],
                extra_cuda_cflags=["-O3", "-DTHOR_SELECT_TUNING"], verbose=False)


def capture(args) -> list[dict]:
    import qsa_check as qc
    from tensorfold.families.qwen4_exp.cuda import thor_qsa

    engine = qc.pp.build_engine(args.model, 4, "int8")
    want = {int(p) for p in args.p0.split(",") if p}
    seen: dict = defaultdict(int)
    slices: list[dict] = []
    orig = thor_qsa.select
    prompt = {"name": ""}

    def hook(scratch, pos0, rows, blocks, warps=None, variant=None):
        p0 = int(pos0[0])
        key = (prompt["name"], p0)
        if p0 in want and seen[key] < args.layers:
            seen[key] += 1
            slices.append({"prompt": prompt["name"], "p0": p0, "layer": seen[key], "rows": int(rows),
                           "blocks": int(blocks), "ratio": scratch.ratio, "budget": scratch.budget, "idw": scratch.idw,
                           "scores": scratch.scores[:rows, :blocks].contiguous().cpu()})
        orig(scratch, pos0, rows, blocks, warps=warps, variant=variant)

    thor_qsa.select = hook
    try:
        for name in [p for p in args.prompts.split(",") if p]:
            n, natural, seed = qc.PROMPTS[name]
            ids = qc.pp.prompt_ids(args.model, n, seed, natural=natural)
            qc.nr.forget(engine)
            prompt["name"] = name
            got = qc.pp.run_prompt(engine, ids)
            print(f"[sel_bench] {name}: {got['tok_s']:.0f} tok/s with the capture inside; "
                  f"{sum(1 for s in slices if s['prompt'] == name)} slices", flush=True)
    finally:
        thor_qsa.select = orig
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return slices


def spread(sc: torch.Tensor, complete: torch.Tensor, b0: int) -> dict:
    """The first digit's spread over a slice's sparse rows (16 sampled): distinct digits a row, and the largest count
    of one digit among 32 consecutive blocks (a warp's lanes in one step), mean and max over the groups."""

    rows = torch.nonzero(complete > 512).squeeze(1)
    rows = rows[torch.linspace(0, rows.numel() - 1, min(16, rows.numel()), device=rows.device).round().long()]
    distinct, mult_mean, mult_max = [], [], []
    for r in rows.tolist():
        c = int(complete[r]) // 32 * 32
        bits = sc[r, :c].view(torch.int32).long() & 0xFFFFFFFF
        key = torch.where(bits >= 2**31, bits ^ 0xFFFFFFFF, bits | 2**31)
        d = (key >> (32 - b0)).view(-1, 32)
        distinct.append(int(torch.unique(d).numel()))
        m = (d[:, :, None] == d[:, None, :]).sum(-1).max(-1).values.float()
        mult_mean.append(float(m.mean()))
        mult_max.append(float(m.max()))
    return {"distinct": sum(distinct) / len(distinct), "mult_mean": sum(mult_mean) / len(mult_mean),
            "mult_max": max(mult_max)}


def bench(args, slices: list[dict]) -> list[dict]:
    from tensorfold.families.qwen4_exp.cuda import attention as attn_mod
    from tensorfold.families.qwen4_exp.cuda import thor_qsa

    ext = tuning_ext()
    dev = "cuda"
    out = []
    for s in slices:
        rows, blocks = s["rows"], s["blocks"]
        sc = s.pop("scores").to(dev)
        ns = SimpleNamespace(scores=sc, nb=blocks, ratio=s["ratio"], budget=s["budget"], idw=s["idw"],
                             ids=torch.zeros((rows, s["idw"]), dtype=torch.int32, device=dev),
                             nk=torch.zeros((rows,), dtype=torch.int32, device=dev),
                             sparse=torch.zeros((rows,), dtype=torch.int32, device=dev))
        pos0 = torch.tensor([s["p0"]], dtype=torch.int32, device=dev)
        top = s["budget"] // s["ratio"]
        attn_mod._launch_select(ns, pos0, rows, blocks)
        ids_ref, nk_ref, sp_ref = ns.ids.clone(), ns.nk.clone(), ns.sparse.clone()
        listed = (sp_ref != 0)[:, None] & (torch.arange(s["idw"], device=dev)[None, :] < nk_ref[:, None])
        complete = (s["p0"] + torch.arange(rows, device=dev) + 1) // s["ratio"]
        res = dict(s)
        res["spread11"] = spread(sc, complete, 11)
        res["spread13"] = spread(sc, complete, 13)
        t: dict = {}
        t["tf"] = timed(lambda: attn_mod._launch_select(ns, pos0, rows, blocks), args.reps)
        thor_qsa.set_sel(False)
        t["triton"] = timed(lambda: thor_qsa.select(ns, pos0, rows, blocks), args.reps)
        equal = {}
        for cfg in CONFIGS:
            for b0 in (11, 13):
                for stop in (0, 1, 2):
                    v = (cfg << 4) | (2 if b0 == 13 else 0) | (stop << 8)
                    name = f"c{cfg}_b{b0}" + (f"_stop{stop}" if stop else "")

                    def call(v=v):
                        ext.select(ns.scores, pos0, ns.ids, ns.nk, ns.sparse, rows, blocks, s["ratio"], top, v)

                    t[name] = timed(call, args.reps)
                    if stop == 0:
                        ns.ids.fill_(-1)
                        ns.nk.fill_(-1)
                        ns.sparse.fill_(-1)
                        call()
                        equal[name] = bool(torch.equal(ns.nk, nk_ref) and torch.equal(ns.sparse, sp_ref)
                                           and torch.equal(ns.ids[listed], ids_ref[listed]))
        res["ms"] = t
        res["equal"] = equal
        full = {k: x for k, x in t.items() if k.startswith("c") and "stop" not in k}
        best = min(full, key=full.get)
        auto = "c" + str(1 if blocks <= 1024 else 2 if blocks <= 2048 else 3 if blocks <= 4096 else 4
                         if blocks <= 8192 else 5 if blocks <= 16384 else 6) + "_b11"
        res["best"], res["auto"] = best, auto
        print(f"[sel_bench] {s['prompt']} p0 {s['p0']} L{s['layer']} blocks {blocks}: tf {t['tf']:.3f} triton "
              f"{t['triton']:.3f} auto {auto} {t[auto]:.3f} (step0 {t[auto + '_stop1']:.3f}, "
              f"steps {t[auto + '_stop2']:.3f})"
              f" best {best} {t[best]:.3f} (step0 {t[best + '_stop1']:.3f}, steps {t[best + '_stop2']:.3f}) ms; "
              f"all equal {all(equal.values())}; d11 {res['spread11']['distinct']:.0f} mult "
              f"{res['spread11']['mult_mean']:.1f}/{res['spread11']['mult_max']:.0f}, d13 "
              f"{res['spread13']['distinct']:.0f} mult {res['spread13']['mult_mean']:.1f}", flush=True)
        out.append(res)
        del sc, ns
        torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--prompts", default="nat32k,syn128k")
    ap.add_argument("--p0", default="2048,7936,16128,32512,65280,130816")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tuning_ext()                                    # built before the weights load
    slices = capture(args)
    res = bench(args, slices)
    (args.out / "sel_bench.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
