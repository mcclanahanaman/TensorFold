"""Cold prefill speed on a running TensorFold server (client only): unique prompts of N tokens, one token out.

    python tools/thor/prefill_ttft.py BASE_URL OUT.json [--tokens 8192,32768] [--repeat 2]

Each prompt is new text (a run id and its own numbers), so nothing resumes from a kept state; the request asks for one
token, so its time is the prompt's fill plus one sampled token. Reports seconds and prompt tok/s (the server's own
prompt count). Every request counts: there is no warm-up exclusion.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from pathlib import Path


def text(tokens: int, seed: int) -> str:
    rng = random.Random(seed)
    words = ["alpha", "bravo", "cobalt", "delta", "ember", "fjord", "garnet", "harbor", "indigo", "juniper", "kelp",
             "lumen", "meadow", "nectar", "onyx", "pylon", "quartz", "raven", "sierra", "tundra"]
    out = [f"Run {seed}."]
    for i in range(tokens * 2 // 3):                       # about 1.5 tokens a word with the numbers
        out.append(words[rng.randrange(len(words))] if i % 3 else str(rng.randrange(100000)))
    return " ".join(out) + "\n\nReply with one word."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("out", type=Path)
    ap.add_argument("--tokens", default="8192,32768")
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()
    base = args.base.rstrip("/")
    with urllib.request.urlopen(base + "/v1/models", timeout=60) as resp:
        model = json.loads(resp.read())["data"][0]["id"]
    rows = []
    seed = int(time.time())
    for n in [int(v) for v in args.tokens.split(",")]:
        for r in range(args.repeat):
            seed += 1
            body = {"model": model, "messages": [{"role": "user", "content": text(n, seed)}], "max_tokens": 1,
                    "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
            req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=3600) as resp:
                result = json.loads(resp.read())
            dt = time.perf_counter() - t0
            usage = result.get("usage", {})
            prompt = int(usage.get("prompt_tokens") or 0)
            cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
            row = {"target": n, "prompt_tokens": prompt, "cached": cached, "seconds": dt,
                   "tok_s": (prompt - cached) / dt if dt > 0 else None}
            rows.append(row)
            print(f"{n}: {prompt} prompt tokens ({cached} cached) in {dt:.2f}s = {row['tok_s']:.0f} tok/s", flush=True)
            args.out.write_text(json.dumps({"base": base, "model": model, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
