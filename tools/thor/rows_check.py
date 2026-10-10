"""In-process check of wider prompt pieces on Jetson Thor: the same prompts through the served
concurrent decoder at each piece size; head logits bit for bit, prompt tok/s and the allocator's peak.

    python tools/thor/rows_check.py MODEL OUT.json [--rows 2048,4096,8192,16384] [--tokens 32768,131072]
        [--parallel 4] [--kv-dtype int8] [--seed 900001]

The engine is built once, as ``tensorfold serve MODEL --backend cuda --kv-dtype KV --parallel N`` builds it. The
engine's own piece size runs on its own decoder; every other size rebuilds the decoder (``prefill_profile.rebuild``:
the same weights, the old buffers freed, the new ones warmed), so each size starts from a fresh decoder and runs the
same prompts in the same order (new text, one token out, nothing resumes). Per prompt: wall tok/s, the first token,
the sha256 of the last pass's head logits (bit for bit) and the allocator's peak above what was allocated before the
prompt (the stream's cache growth and the passes' temporaries: the pieces' buffers are allocated with the decoder).
``same_logits_as_first`` compares each prompt's logits with the first size's.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import prefill_profile as pp  # noqa: E402

GIB = 2**30


def one_size(engine, rows: int, prompts: dict, fresh: bool) -> dict:
    """Run every prompt with ``rows``-row pieces (``fresh``: the engine's own decoder has run nothing yet)."""

    if fresh and engine.multi.prefill_rows == rows:
        room = {"rebuilt": False, "gate_room_gib": engine.multi.memory_gate.room / GIB}
    else:
        room = {"rebuilt": True, **pp.rebuild(engine, rows)}
    entry = {"rows": rows, **room, "runs": []}
    for n, ids in prompts.items():
        pp.L.reset("plain")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        got = pp.run_prompt(engine, ids)
        got["peak_above_gib"] = (torch.cuda.max_memory_allocated() - before) / GIB
        entry["runs"].append({"target": n, **got})
        print(f"[rows] {rows} rows, {n} tokens: {got['seconds']:.2f}s = {got['tok_s']:.0f} tok/s, first "
              f"{got['first']}, logits {got['logits_sha']}, peak +{got['peak_above_gib']:.2f} GiB"
              + (f", error {got['error']}" if got["error"] else ""), flush=True)
    return entry


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Wider prompt pieces in process: logits, tok/s, peak memory.")
    ap.add_argument("model", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--rows", default="2048,4096,8192,16384")
    ap.add_argument("--tokens", default="32768,131072")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--kv-dtype", default="int8")
    ap.add_argument("--seed", type=int, default=900001)
    args = ap.parse_args(argv)
    sizes = [int(v) for v in args.rows.split(",") if v]
    lengths = [int(v) for v in args.tokens.split(",") if v]
    t0 = time.perf_counter()
    engine = pp.build_engine(args.model, args.parallel, args.kv_dtype)
    own = engine.multi.prefill_rows
    print(f"[rows] engine up in {time.perf_counter() - t0:.0f}s with {own}-row pieces; gate room "
          f"{engine.multi.memory_gate.room / GIB:.2f} GiB", flush=True)
    prompts = {n: pp.prompt_ids(args.model, n, args.seed + i) for i, n in enumerate(lengths)}
    report = {"model": str(args.model), "parallel": args.parallel, "kv_dtype": args.kv_dtype, "engine_rows": own,
              "torch": torch.__version__, "sizes": []}
    for i, rows in enumerate(sizes):
        entry = one_size(engine, rows, prompts, fresh=i == 0)
        base = report["sizes"][0]["runs"] if report["sizes"] else entry["runs"]
        entry["same_logits_as_first"] = [a["logits_sha"] == b["logits_sha"] for a, b in zip(entry["runs"], base)]
        report["sizes"].append(entry)
        args.out.write_text(json.dumps(report, indent=1))
    same = all(all(e["same_logits_as_first"]) for e in report["sizes"])
    print(f"[rows] done in {time.perf_counter() - t0:.0f}s; head logits {'bit-identical' if same else 'DIFFER'} "
          f"across {','.join(map(str, sizes))}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
