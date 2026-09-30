"""Decode-sized expert calls under CUDA graphs: TensorFold's nvfp4_expert_kernel (decode plan, fp32 down, as the
engine runs it) against FlashInfer's CuTe-DSL W4A16 grouped GEMMs, GPU time only (no host launch cost).

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


def graph_time(fn, reps: int) -> float:
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
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="/out/decode_graph.json")
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--ms", default="1,2,4,8,16,32,64")
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()
    rep = {"rows": {}, "errors": {}}

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
        plan = grouped.Plan(m, mc.TOP + 1, mc.E + 1, "cuda", prefill=False)
        act = torch.zeros(m * (mc.TOP + 1), mc.NI, dtype=torch.bfloat16, device="cuda")
        y = torch.zeros(m * (mc.TOP + 1), mc.D, dtype=torch.float32, device="cuda")

        def tf_call():
            grouped.route(picks, plan)
            nvx.gate_up(x, ex, plan, act, m, skip=mc.E)
            nvx.down(act, ex, plan, y, m, skip=mc.E)

        try:
            row["tf_ms"] = graph_time(tf_call, args.reps)
        except Exception as exc:  # noqa: BLE001
            rep["errors"][f"M{m}.tf"] = repr(exc)[:500]
        best = None
        for tac in fw.sweep_tactics(m):
            fi.tactic = tac
            try:
                prep = fi.prepare(x, ids)
                t = graph_time(lambda: fi.gemms(prep), args.reps)
            except Exception as exc:  # noqa: BLE001
                rep["errors"][f"M{m}.fi.{tac}"] = repr(exc)[:300]
                continue
            if best is None or t < best[1]:
                best = (tac, t)
        if best is not None:
            row["fi_ms"], row["fi_tactic"] = best[1], best[0]
        rep["rows"][f"M{m}"] = row
        print(f"[{mc.stamp()}] M={m}: TensorFold {row.get('tf_ms', float('nan')):.4f} ms, "
              f"FlashInfer W4A16 {row.get('fi_ms', float('nan')):.4f} ms {row.get('fi_tactic')}", flush=True)
        Path(args.out).write_text(json.dumps(rep, indent=1, default=str))
    Path(args.out).write_text(json.dumps(rep, indent=1, default=str))
    print("errors:", list(rep["errors"])[:10], flush=True)


if __name__ == "__main__":
    main()
