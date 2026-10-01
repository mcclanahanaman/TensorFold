"""TensorFold's exactness invariants on a running server (client only), for the Thor experts' G-B.

    python tools/thor/thor_invariants.py BASE_URL OUT.json [--tokens 48] [--parallel 4] [--long 3000]

Greedy requests with each token's id and top-5 log probabilities; every comparison is on both:
  solo == concurrent   each prompt alone, then all of them at once (``--parallel`` requests in flight)
  drafted == serial    the default (MTP drafts) against ``"draft": false``
  resend == fresh      an identical second request (resumed from the kept prompt state) against the first
``--long N`` (THOR-K4) repeats them on prompts of about N tokens, which span prompt passes and many 64-row chunks:
drafted == serial and resend == fresh on four prompts; each one's next turn (its answer, then a new question), drafted
and resumed from the first turn's kept state, against the same turn with ``"draft": false`` (which never resumes); and
four new prompts sent at once (fresh, packed into shared passes) against each alone with ``"draft": false``.
Exit status 0 when every comparison holds.
"""

from __future__ import annotations

import argparse
import json
import random
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


def request(base: str, model: str, prompt, tokens: int, **extra) -> dict:
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    payload = {"model": model, "messages": messages, "max_tokens": tokens,
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
            "cached": (result.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
            "prompt_tokens": result.get("usage", {}).get("prompt_tokens"),
            "text": result["choices"][0].get("message", {}).get("content")}


WORDS = ("alpha", "bravo", "cobalt", "delta", "ember", "fjord", "garnet", "harbor", "indigo", "juniper", "kelp",
         "lumen", "meadow", "nectar", "onyx", "pylon", "quartz", "raven", "sierra", "tundra")


def long_text(seed: int, tokens: int) -> str:
    """About ``tokens`` tokens of words and numbers (about 3.1 tokens a word in Flash Next's tokenizer), then a task."""

    rng = random.Random(seed)
    words = [WORDS[rng.randrange(len(WORDS))] if i % 3 else str(rng.randrange(100000))
             for i in range(tokens * 10 // 31)]
    return f"Text {seed}: " + " ".join(words) + "\n\nHow many times does the word alpha appear above? Explain briefly."


def same(a: dict, b: dict) -> bool:
    return a["ids"] == b["ids"] and a["logprobs"] == b["logprobs"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("out", type=Path)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--long", type=int, default=0, help="also prompts of about this many tokens, and their next turns")
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
    report = {"base": base, "model": model, "tokens": args.tokens, "checks": checks,
              "resend_cached_tokens": [r["cached"] for r in resend],
              "solo": solo, "together": together, "serial": serial, "resend": resend}
    if args.long:                  # each check's runs start fresh or resume as it says (kept states are slots)
        texts = [long_text(seed, args.long) for seed in range(len(prompts))]
        lsolo = [request(base, model, t, args.tokens) for t in texts]                     # fresh, one at a time
        lserial = [request(base, model, t, args.tokens, draft=False) for t in texts]      # never resumes
        lresend = [request(base, model, t, args.tokens) for t in texts]                   # resumes its kept state
        turns = [[{"role": "user", "content": t}, {"role": "assistant", "content": a["text"] or ""},
                  {"role": "user", "content": "Now list the last three numbers above, nothing else."}]
                 for t, a in zip(texts, lsolo)]
        lnext = [request(base, model, m, args.tokens) for m in turns]                    # resumes the first turn's
        lnext_serial = [request(base, model, m, args.tokens, draft=False) for m in turns]
        fresh = [long_text(100 + seed, args.long) for seed in range(len(prompts))]
        with ThreadPoolExecutor(max_workers=len(fresh)) as pool:                          # fresh, packed together
            ltogether = list(pool.map(lambda t: request(base, model, t, args.tokens), fresh))
        lalone = [request(base, model, t, args.tokens, draft=False) for t in fresh]
        checks.update({"long drafted == serial": [same(a, b) for a, b in zip(lsolo, lserial)],
                       "long resend == fresh": [same(a, b) for a, b in zip(lsolo, lresend)],
                       "long next turn (resumed) == serial": [same(a, b) for a, b in zip(lnext, lnext_serial)],
                       "long concurrent == solo (serial)": [same(a, b) for a, b in zip(ltogether, lalone)]})
        report.update({"long_prompt_tokens": [r["prompt_tokens"] for r in lsolo],
                       "long_resend_cached_tokens": [r["cached"] for r in lresend],
                       "long_next_turn_prompt_tokens": [r["prompt_tokens"] for r in lnext],
                       "long_next_turn_cached_tokens": [r["cached"] for r in lnext],
                       "long_together_cached_tokens": [r["cached"] for r in ltogether],
                       "long_solo": lsolo, "long_serial": lserial, "long_resend": lresend, "long_next": lnext,
                       "long_next_serial": lnext_serial, "long_together": ltogether, "long_alone": lalone})
    ok = all(all(v) for v in checks.values())
    report["pass"] = ok
    args.out.write_text(json.dumps(report, indent=1))
    for name, values in checks.items():
        print(f"{name}: {'PASS' if all(values) else 'FAIL'} {values}", flush=True)
    print(f"resend cached prompt tokens: {report['resend_cached_tokens']}", flush=True)
    if args.long:
        print(f"long prompts {report['long_prompt_tokens']} tokens; resend cached "
              f"{report['long_resend_cached_tokens']};"
              f" next turns {report['long_next_turn_prompt_tokens']} tokens, cached "
              f"{report['long_next_turn_cached_tokens']}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
