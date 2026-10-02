"""Serving A/B on Jetson Thor (FR-OPS-17): one client pass over one running TensorFold server. Every server gets the
same requests in the same order, so fresh servers of different builds (stock TensorFold, the Thor stack with its
switches off or on) compare on the same prompts in one session.

    python tools/thor/serve_ab.py BASE_URL OUT.json [--label NAME] [--seed S] [--prefill 8192,32768,131072]
        [--repeat 2] [--followups 16384,32768,61440] [--decode 256] [--streams 3]

In this order, every request counted (no warm-up exclusion), thinking off, greedy:
- prefill: for each --prefill length, --repeat new synthetic prompts (prefill_ttft.py's text; one --seed gives the same
  prompts on every server), one token out, streamed: TTFT (the first event after the opening role chunk) and prompt
  tok/s (the server's prompt count, less its cached tokens, over TTFT);
- follow-ups: for each --followups length, a chat of one long user message (new synthetic text) answered with up to 24
  tokens, then the next turn (that message, the reply as the server sent it, a short new user message), one token out:
  the next turn's TTFT and the server's cached-token count (where it resumed: TensorFold keeps a prompt's state at its
  last assistant start, which the next turn shares);
- decode: one stream (a short prompt, --decode tokens, end tokens ignored), then --streams at once (other prompts): each
  stream's decode tok/s from the server's figures (tokens after the first over its decode time), and all
  streams' tokens after their first over the wall time from the earliest first token to the last end; each stream's
  MTP acceptance and tokens a round (the server's ``speculative`` figures), which near-tie flips can move.
Each reply's token hash (the server's ``token_sha``) is kept: two builds gave the same greedy text where they match.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prefill_ttft import text  # noqa: E402

FOLLOW_REPLY = 24
NEXT_TURN = "Thanks. Now answer in one short sentence: which number appeared first in my message?"
DECODE_PROMPTS = (
    "Write a detailed essay on the history of the printing press, from Gutenberg to the digital age.",
    "Explain, step by step and with examples, how a compiler turns source code into machine code.",
    "Describe the water cycle in depth, covering evaporation, condensation, precipitation and runoff.",
    "Tell a long story about a lighthouse keeper who finds a map in a bottle.",
)


def model_id(base: str) -> str:
    with urllib.request.urlopen(base + "/v1/models", timeout=60) as resp:
        return json.loads(resp.read())["data"][0]["id"]


def chat(base: str, model: str, messages: list[dict], max_tokens: int, *, ignore_eos: bool = False,
         timeout: float = 3600.0) -> dict:
    """One streamed chat request: client TTFT and wall time, the server's usage and runtime figures, the reply text."""

    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    if ignore_eos:
        body["ignore_eos"] = True
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Accept": "text/event-stream"})
    t0 = time.perf_counter()
    first = None
    usage, stats, spec, pieces = {}, {}, {}, []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(f"the server stream ended with an error: {event['error']}")
            choice = (event.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            opening = set(delta) <= {"role"} and choice.get("finish_reason") is None and "usage" not in event
            if first is None and not opening:
                first = time.perf_counter()
            pieces.append(delta.get("content") or delta.get("reasoning_content") or "")
            usage = event.get("usage") or usage
            stats = event.get("tensorfold") or stats
            spec = event.get("speculative") or spec
    end = time.perf_counter()
    first = first if first is not None else end
    prompt = int(usage.get("prompt_tokens") or 0)
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    n = int(usage.get("completion_tokens") or 0)
    ttft = first - t0
    # the CUDA server's stats block (prefill_s, decode_s, rounds, drafted, accepted), or the Mac server's runtime and
    # speculative blocks
    decode_s, rounds, drafted = stats.get("decode_s"), stats.get("rounds"), stats.get("drafted")
    decode = stats.get("tokens_per_second") if decode_s is None else (n - 1) / decode_s if decode_s else None
    acceptance = spec.get("acceptance_rate") if drafted is None else stats.get("accepted", 0) / drafted if drafted \
        else None
    per_round = spec.get("tokens_per_round") if rounds is None else (n - 1) / rounds if rounds else None
    return {"t0": t0, "first": first, "end": end, "ttft_s": round(ttft, 4), "total_s": round(end - t0, 4),
            "prompt_tokens": prompt, "cached": cached, "completion_tokens": n,
            "tok_s": round((prompt - cached) / ttft, 1) if prompt and ttft > 0 else None,
            "server_prefill_s": stats.get("prefill_s", stats.get("prefill_seconds")), "server_decode_tok_s": decode,
            "token_sha": stats.get("token_sha"), "drafts": stats.get("drafts"), "acceptance": acceptance,
            "tokens_per_round": per_round, "text": "".join(pieces)}


def public(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in ("t0", "first", "end")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("out", type=Path)
    ap.add_argument("--label", default="")
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--prefill", default="8192,32768,131072")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--followups", default="16384,32768,61440")
    ap.add_argument("--decode", type=int, default=256)
    ap.add_argument("--streams", type=int, default=3)
    args = ap.parse_args()
    base = args.base.rstrip("/")
    model = model_id(base)
    report: dict = {"base": base, "model": model, "label": args.label, "seed": args.seed, "prefill": [],
                    "followups": [], "decode": {}}
    seed = args.seed
    tag = f"[ab {args.label}]" if args.label else "[ab]"

    def save() -> None:
        args.out.write_text(json.dumps(report, indent=1))

    for n in [int(v) for v in args.prefill.split(",") if v]:
        for r in range(args.repeat):
            seed += 1
            got = chat(base, model, [{"role": "user", "content": text(n, seed)}], 1)
            report["prefill"].append({"target": n, "round": r, "seed": seed, **public(got)})
            print(f"{tag} prefill {n} #{r}: {got['prompt_tokens']} tokens ({got['cached']} cached), TTFT "
                  f"{got['ttft_s']:.3f} s = {got['tok_s']} tok/s; sha {got['token_sha']}", flush=True)
            save()
    for n in [int(v) for v in args.followups.split(",") if v]:
        seed += 1
        first = [{"role": "user", "content": text(n, seed)}]
        one = chat(base, model, first, FOLLOW_REPLY)
        nxt = first + [{"role": "assistant", "content": one["text"]}, {"role": "user", "content": NEXT_TURN}]
        two = chat(base, model, nxt, 1)
        report["followups"].append({"target": n, "seed": seed, "first": public(one), "next": public(two)})
        print(f"{tag} follow-up {n}: first turn {one['prompt_tokens']} tokens in {one['ttft_s']:.3f} s TTFT; next "
              f"turn {two['prompt_tokens']} tokens, {two['cached']} cached ({two['prompt_tokens'] - two['cached']} "
              f"filled), TTFT {two['ttft_s']:.3f} s; reply {one['text'][:30]!r}", flush=True)
        save()
    one = chat(base, model, [{"role": "user", "content": DECODE_PROMPTS[0]}], args.decode, ignore_eos=True)
    report["decode"]["one"] = public(one)
    print(f"{tag} decode 1 stream: {one['completion_tokens']} tokens, server {one['server_decode_tok_s'] or 0:.1f} "
          f"tok/s, client {(one['completion_tokens'] - 1) / (one['end'] - one['first']):.1f} tok/s after the first; "
          f"MTP acceptance {one['acceptance']}, {one['tokens_per_round']} tokens a round; sha {one['token_sha']}",
          flush=True)
    rows: list = [None] * args.streams

    def go(i: int) -> None:
        rows[i] = chat(base, model, [{"role": "user", "content": DECODE_PROMPTS[1 + i % (len(DECODE_PROMPTS) - 1)]}],
                       args.decode, ignore_eos=True)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(args.streams)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if any(r is None for r in rows):
        raise SystemExit("a decode stream failed")
    wall = max(r["end"] for r in rows) - min(r["first"] for r in rows)
    agg = sum(r["completion_tokens"] - 1 for r in rows) / wall if wall > 0 else None
    report["decode"]["many"] = {"streams": [public(r) for r in rows], "wall_tok_s": agg,
                                "server_sum_tok_s": sum(r["server_decode_tok_s"] or 0 for r in rows)}
    each = " / ".join(f"{r['server_decode_tok_s'] or 0:.1f}" for r in rows)
    accepted = " / ".join(f"{r['acceptance'] or 0:.3f}" for r in rows)
    print(f"{tag} decode {args.streams} streams: server {each} tok/s (sum "
          f"{report['decode']['many']['server_sum_tok_s']:.1f}), wall {agg:.1f} tok/s; MTP acceptance {accepted}",
          flush=True)
    save()


if __name__ == "__main__":
    main()
