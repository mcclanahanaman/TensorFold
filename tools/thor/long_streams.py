"""Concurrent long-prompt streams on one running TensorFold server (FR-OPS-17, THOR-DEC-1 full-window stress): every
--lengths entry is one chat whose single user message is prefill_ttft.text(length, seed + i), all sent at once, greedy,
end tokens ignored, so the streams' caches grow together toward the window.

    python tools/thor/long_streams.py URL OUT.json [--label L] [--lengths 240000,220000,200000,180000]
        [--seed 20261003] [--max-tokens 256] [--timeout 7200]

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prefill_ttft import text  # noqa: E402
from serve_ab import chat, model_id, public  # noqa: E402


def _lengths(raw: str) -> list[int]:
    return [int(v) for v in raw.split(",") if v]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("out", type=Path)
    ap.add_argument("--label", default="")
    ap.add_argument("--lengths", type=_lengths, default=[240000, 220000, 200000, 180000])
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=7200.0)
    return ap.parse_args(argv)


def run(url: str, lengths: list[int], *, seed: int, max_tokens: int, timeout: float, label: str = "") -> dict:
    base = url.rstrip("/")
    model = model_id(base)
    rows: list = [None] * len(lengths)

    def go(i: int, n: int) -> None:
        try:
            got = chat(base, model, [{"role": "user", "content": text(n, seed + i)}], max_tokens,
                       ignore_eos=True, timeout=timeout)
        except Exception as exc:  # a failed stream is recorded in its row, never raised
            rows[i] = {"i": i, "target": n, "error": repr(exc)[:400]}
        else:
            rows[i] = {"i": i, "target": n, **public(got)}

    before = time.perf_counter()
    threads = [threading.Thread(target=go, args=(i, n)) for i, n in enumerate(lengths)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_s = round(time.perf_counter() - before, 3)
    return {"label": label, "url": base, "model": model, "lengths": list(lengths), "seed": seed,
            "max_tokens": max_tokens, "streams": rows, "wall_s": wall_s,
            "errors": sum(1 for r in rows if "error" in r)}


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    rep = run(a.url, a.lengths, seed=a.seed, max_tokens=a.max_tokens, timeout=a.timeout, label=a.label)
    for s in rep["streams"]:
        if "error" in s:
            print(f"stream {s['i']} target {s['target']}: ERROR {s['error']}", flush=True)
        else:
            print(f"stream {s['i']} target {s['target']}: prompt {s['prompt_tokens']} ttft {s['ttft_s']}s out "
                  f"{s['completion_tokens']} decode {s['server_decode_tok_s']} tok/s", flush=True)
    print(f"wall {rep['wall_s']}s errors {rep['errors']}", flush=True)
    a.out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return 1 if rep["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
