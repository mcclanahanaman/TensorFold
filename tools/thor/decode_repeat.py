"""Repeated decode requests on one running TensorFold server (THOR-K3): the same greedy requests several
times, one stream and then several at once, so rounds that replay recorded CUDA graphs can be told from the first
requests' rounds (which record them). Every request is kept; wall-clock request times let a server log's timestamped
events be matched to requests.

    python tools/thor/decode_repeat.py BASE_URL OUT.json [--label NAME] [--repeat 3] [--decode 256] [--streams 1,4]

For each --streams count n and each repeat, n requests at once (serve_ab.py's decode prompts, end tokens ignored,
thinking off): each stream's server decode tok/s, tokens a round, MTP acceptance and token hash, and the batch's wall
tokens after the first over the wall time. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from serve_ab import DECODE_PROMPTS, chat, model_id  # noqa: E402


def batch(base: str, model: str, n: int, tokens: int) -> dict:
    out: list[dict | None] = [None] * n
    t0 = time.time()

    def go(i: int) -> None:
        msgs = [{"role": "user", "content": DECODE_PROMPTS[i % len(DECODE_PROMPTS)]}]
        r = chat(base, model, msgs, tokens, ignore_eos=True)
        r["wall_start"], r["wall_end"] = r["t0"] - t0, r["end"] - t0
        r.pop("text", None)
        out[i] = r

    threads = [threading.Thread(target=go, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = [r for r in out if r is not None]
    after = sum(max(0, r["completion_tokens"] - 1) for r in rows)
    span = max(r["end"] for r in rows) - min(r["first"] for r in rows) if rows else 0.0
    return {"streams": n, "t_start": t0, "t_end": time.time(), "rows": rows,
            "wall_tok_s": round(after / span, 2) if span > 0 else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("out", type=Path)
    ap.add_argument("--label", default="")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--decode", type=int, default=256)
    ap.add_argument("--streams", default="1,4")
    a = ap.parse_args(argv)
    base = a.base.rstrip("/")
    model = model_id(base)
    runs = []
    for n in [int(x) for x in a.streams.split(",") if x]:
        for k in range(a.repeat):
            b = batch(base, model, n, a.decode)
            b["repeat"] = k
            runs.append(b)
            each = " / ".join(f"{r['server_decode_tok_s']:.1f}" if r["server_decode_tok_s"] else "-" for r in b["rows"])
            tpr = " / ".join(f"{r['tokens_per_round']:.2f}" if r["tokens_per_round"] else "-" for r in b["rows"])
            shas = ",".join(str(r["token_sha"])[:8] for r in b["rows"])
            print(f"[rep {a.label}] {n} stream(s) #{k}: server {each} tok/s; wall {b['wall_tok_s']} tok/s; "
                  f"tokens a round {tpr}; sha {shas}", flush=True)
    a.out.write_text(json.dumps({"label": a.label, "runs": runs}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
