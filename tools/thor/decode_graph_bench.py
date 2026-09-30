"""Expert calls under CUDA graphs: TensorFold's nvfp4_expert_kernel (decode plan and fp32 down up to 64 rows,
the prompt plan and bf16 down above, as the engine runs them) against FlashInfer's CuTe-DSL W4A16 grouped GEMMs,
GPU time only (no host launch cost), interleaved rounds after a GPU warm-up, medians reported.

    FLASHINFER_NO_DOWNLOAD=1 python tools/thor/decode_graph_bench.py --model /hf/... --out /out/decode_graph.json

For each M (tokens a step: 1 plain decode, 2-4 with drafts, 8-64 several streams) one graph replays the whole call
(TensorFold: plan + gate_up + down; FlashInfer: both GEMMs on metadata made before capture, every tactic tried) and
CUDA events time --reps replays.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import moe_common as mc  # noqa: E402


def capture(fn) -> torch.cuda.CUDAGraph:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    return g


def replay_ms(g: torch.cuda.CUDAGraph, reps: int) -> float:
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / reps


_BURN = {}


def burn(seconds: float = 0.3) -> None:
    """Keep the GPU busy for a moment so its clock governor has ramped up before a timing block."""

    if "a" not in _BURN:
        _BURN["a"] = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    a = _BURN["a"]
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    while True:
        for _ in range(8):
            a @ a
        e1.record()
        e1.synchronize()
        if e0.elapsed_time(e1) > seconds * 1e3:
            break


def interleaved(graphs: dict, reps: int, rounds: int) -> dict:
    """Median over ``rounds`` of each graph's replay time, the order alternating, a GPU burn before each round."""

    times = {k: [] for k in graphs}
    keys = list(graphs)
    for r in range(rounds):
        burn()
        for k in (keys if r % 2 == 0 else keys[::-1]):
            times[k].append(replay_ms(graphs[k], reps))
    return {k: sorted(v)[len(v) // 2] for k, v in times.items()} | {k + "_all": v for k, v in times.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="/out/decode_graph.json")
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--ms", default="1,2,4,8,16,32,64,512,2048,4096,8192")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--tf-decode-plan", action="store_true", help="0.3.x decode plan + fp32 slots at <= 64 rows")
    args = ap.parse_args()
    rep = {"rows": {}, "errors": {}, "args": vars(args)}

    from tensorfold.cuda import experts as grouped
    from tensorfold.cuda.nvfp4 import experts as nvx

    import fi_w4a16_moe as fw

    layer = mc.Checkpoint(args.model).layer(args.layer)
    p = lambda q: (q.w, q.s, q.s2)  # noqa: E731
    ex = nvx.make(p(layer.gate), p(layer.up), p(layer.down))
    fi = fw.FlashInferW4A16(layer)
    for m in [int(v) for v in args.ms.split(",")]:
        x = mc.activations(m, seed=3000 + m)
        ids, w = mc.route(x, layer.router)
        row = {}
        picks = torch.cat([ids, torch.full((m, 1), mc.E, dtype=torch.int32, device="cuda")], 1).contiguous()
        # 0.6.0: decode windows run the prompt arithmetic too (Buffers(moe_prefill=True)): prompt plan, bf16 slots;
        # --tf-decode-plan keeps 0.3.x's decode plan and fp32 slots at <= 64 rows
        prefill = m > 64 or not args.tf_decode_plan
        plan = grouped.Plan(m, mc.TOP + 1, mc.E + 1, "cuda", prefill=prefill)
        tile = getattr(nvx, "PREFILL_TILE", grouped.PREFILL_TILE)
        act = torch.zeros(m * (mc.TOP + 1), mc.NI, dtype=torch.bfloat16, device="cuda")
        y = torch.zeros(m * (mc.TOP + 1), mc.D, dtype=torch.bfloat16 if prefill else torch.float32, device="cuda")

        def tf_call():
            grouped.route(picks, plan, tile)
            nvx.gate_up(x, ex, plan, act, m, skip=mc.E)
            nvx.down(act, ex, plan, y, m, skip=mc.E)

        graphs = {}
        try:
            graphs["tf"] = capture(tf_call)
        except Exception as exc:  # noqa: BLE001
            rep["errors"][f"M{m}.tf"] = repr(exc)[:500]
        best = None
        for tac in fw.sweep_tactics(m):
            fi.tactic = tac
            try:
                prep = fi.prepare(x, ids)
                g = capture(lambda: fi.gemms(prep))
                burn(0.1)
                t = replay_ms(g, args.reps)
            except Exception as exc:  # noqa: BLE001
                rep["errors"][f"M{m}.fi.{tac}"] = repr(exc)[:300]
                continue
            if best is None or t < best[1]:
                best = (tac, t, g, prep)
        if best is not None:
            graphs["fi"] = best[2]
            row["fi_tactic"] = best[0]
        res = interleaved(graphs, args.reps, rounds=7)
        row["tf_ms"], row["fi_ms"] = res.get("tf"), res.get("fi")
        row["tf_all"], row["fi_all"] = res.get("tf_all"), res.get("fi_all")
        rep["rows"][f"M{m}"] = row
        print(f"[{mc.stamp()}] M={m}: TensorFold {row.get('tf_ms') or float('nan'):.4f} ms "
              f"{[round(v, 4) for v in row.get('tf_all') or []]}, FlashInfer W4A16 "
              f"{row.get('fi_ms') or float('nan'):.4f} ms {[round(v, 4) for v in row.get('fi_all') or []]} "
              f"{row.get('fi_tactic')}", flush=True)
        Path(args.out).write_text(json.dumps(rep, indent=1, default=str))
    Path(args.out).write_text(json.dumps(rep, indent=1, default=str))
    print("errors:", list(rep["errors"])[:10], flush=True)


if __name__ == "__main__":
    main()
