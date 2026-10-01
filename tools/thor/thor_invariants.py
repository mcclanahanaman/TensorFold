"""TensorFold's exactness invariants on a running server (client only), for the Thor experts' G-B.

    python tools/thor/thor_invariants.py BASE_URL OUT.json [--tokens 48] [--parallel 4]

Greedy requests with each token's id and top-5 log probabilities; every comparison is on both:
  solo == concurrent   each prompt alone, then all of them at once (``--parallel`` requests in flight)
  drafted == serial    the default (MTP drafts) against ``"draft": false``
  resend == fresh      an identical second request (resumed from the kept prompt state) against the first
Exit status 0 when every comparison holds.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROMPTS = (
    "Write a Python function that merges two sorted lists, then explain its complexity.",
    "Explain why the sky is blue in simple language.",
    "Describe a calm morning beside a lake in five sentences.",
    "List the first ten prime numbers and say what makes 2 special.",
)


def request(base: str, model: str, prompt: str, tokens: int, **extra) -> dict:
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens,
               "temperature": 0, "logprobs": True, "top_logprobs": 5, "return_token_ids": True,
               "chat_template_kwargs": {"enable_thinking": False}, **extra}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as resp:
        result = json.loads(resp.read())
    content = (result["choices"][0].get("logprobs") or {}).get("content") or []
    return {"ids": result.get("tensorfold", {}).get("token_ids"),
            "logprobs": [(c.get("logprob"), [(t.get("token"), t.get("logprob")) for t in c.get("top_logprobs") or []])
                         for c in content],
            "cached": (result.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens")}


def same(a: dict, b: dict) -> bool:
    return a["ids"] == b["ids"] and a["logprobs"] == b["logprobs"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("out", type=Path)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--parallel", type=int, default=4)
    args = ap.parse_args()
    base = args.base.rstrip("/")
    with urllib.request.urlopen(base + "/v1/models", timeout=60) as resp:
        model = json.loads(resp.read())["data"][0]["id"]
    prompts = PROMPTS[:args.parallel]
    solo = [request(base, model, p, args.tokens) for p in prompts]
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        together = list(pool.map(lambda p: request(base, model, p, args.tokens), prompts))
    serial = [request(base, model, p, args.tokens, draft=False) for p in prompts]
    resend = [request(base, model, p, args.tokens) for p in prompts]
    checks = {"solo == concurrent": [same(a, b) for a, b in zip(solo, together)],
              "drafted == serial": [same(a, b) for a, b in zip(solo, serial)],
              "resend == fresh": [same(a, b) for a, b in zip(solo, resend)]}
    ok = all(all(v) for v in checks.values())
    report = {"base": base, "model": model, "tokens": args.tokens, "checks": checks, "pass": ok,
              "resend_cached_tokens": [r["cached"] for r in resend],
              "solo": solo, "together": together, "serial": serial, "resend": resend}
    args.out.write_text(json.dumps(report, indent=1))
    for name, values in checks.items():
        print(f"{name}: {'PASS' if all(values) else 'FAIL'} {values}", flush=True)
    print(f"resend cached prompt tokens: {report['resend_cached_tokens']}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
