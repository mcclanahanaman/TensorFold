"""Agentic follow-ups on one running TensorFold server (FR-OPS-17): for each --lengths target, a chat that opens with
one long user message (prefill_ttft.py's synthetic text), then --turns follow-up turns, each adding the assistant's
reply as the server sent it and a short new user message (a tool result of --tool-tokens synthetic tokens); each turn's
TTFT, prompt tokens, cached tokens (where the server resumed), filled tokens (prompt - cached) and reply hash; every
request counted, thinking off, greedy.

    python tools/thor/followup_ttft.py BASE_URL OUT.json [--label NAME] [--seed S] [--lengths 16384,32768]
        [--turns 4] [--tool-tokens 300] [--reply 24]

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prefill_ttft import text  # noqa: E402
from serve_ab import chat, model_id, public  # noqa: E402

TOOL_PREFIX = "Tool result:\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("out", type=Path)
    ap.add_argument("--label", default="")
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--lengths", default="16384,32768")
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--tool-tokens", type=int, default=300)
    ap.add_argument("--reply", type=int, default=24)
    args = ap.parse_args(argv)
    base = args.base.rstrip("/")
    model = model_id(base)
    lengths = [int(v) for v in args.lengths.split(",") if v]
    report = {"base": base, "model": model, "label": args.label, "seed": args.seed, "lengths": lengths,
              "turns": args.turns, "tool_tokens": args.tool_tokens, "reply": args.reply, "conversations": []}

    def save() -> None:
        args.out.write_text(json.dumps(report, indent=1))

    tag = f"[followup {args.label}]" if args.label else "[followup]"
    seed = args.seed
    for n in lengths:
        seed += 1
        messages = [{"role": "user", "content": text(n, seed)}]
        conv = {"target": n, "seed": seed, "turns": []}
        report["conversations"].append(conv)
        for t in range(args.turns + 1):
            if t > 0:
                messages = messages + [{"role": "assistant", "content": last["text"]},
                                       {"role": "user", "content": TOOL_PREFIX + text(args.tool_tokens,
                                                                                      seed * 1000 + t)}]
            last = chat(base, model, messages, args.reply)
            row = {"turn": t, **public(last)}
            row["filled"] = row["prompt_tokens"] - row["cached"]
            conv["turns"].append(row)
            print(f"{tag} {n} turn {t}: {row['prompt_tokens']} tokens, {row['cached']} cached "
                  f"({row['filled']} filled), TTFT {row['ttft_s']:.3f} s; sha {row['token_sha']}", flush=True)
            save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
