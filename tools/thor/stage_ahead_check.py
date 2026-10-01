"""Stage-ahead A/B on Jetson Thor (FR-OPS-17): one engine, the same prompts with TENSORFOLD_THOR_STAGE_AHEAD=1 and =0.

    python tools/thor/stage_ahead_check.py MODEL_DIR OUT_DIR

The engine reads the switch at each ``stage`` call, so both settings run in one process on the same prompts: cold
prefill wall time and head-logit bits for synthetic 8k/32k/128k, natural-text 32k warm and with the n-gram table
dropped (``ngram_residency.Table.drop``), and 1-user greedy decode (512 tokens, interleaved, three of each). Writes
stage_ahead.json and ledger_metrics.json (prefill_* and decode_1 with the switch on, the identity checks).
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ngram_residency as nr  # noqa: E402
import prefill_profile as pp  # noqa: E402

FLAG = "TENSORFOLD_THOR_STAGE_AHEAD"
CHAT = "Write a Python function that merges two sorted lists, then explain its complexity and give two test cases."


def chat_ids(model: Path, text: str) -> list[int]:
    from tokenizers import Tokenizer

    from tensorfold.cuda.chat_template import ChatTemplate

    rendered = ChatTemplate(model).render([{"role": "user", "content": text}], tools=None, enable_thinking=False)
    return Tokenizer.from_file(str(model / "tokenizer.json")).encode(rendered, add_special_tokens=False).ids


def decode(engine, ids: list[int], count: int) -> dict:
    """The server's path for one greedy request of ``count`` tokens (end tokens ignored): decode tok/s and the ids."""

    from tensorfold.cuda.streams import Stream

    multi = engine.multi
    nr.forget(engine)
    s = Stream(list(ids), count, None, draft=True, stop_eos=False)
    t0 = time.perf_counter()
    multi.admit(s)
    while not s.done:
        multi.finish(multi.round())
    total = time.perf_counter() - t0
    st = s.stats()
    return {"tokens": len(s.out), "decode_s": st["decode_s"], "total_s": round(total, 3),
            "tok_s": round((len(s.out) - 1) / st["decode_s"], 2), "accepted": st["accepted"], "rounds": st["rounds"],
            "ids_sha": hashlib.sha256(",".join(map(str, s.out)).encode()).hexdigest()[:16]}


def main() -> None:
    model, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    engine = pp.build_engine(model, 4, "int8")
    table = nr.Table(engine.w)
    runs: list = []
    report: dict = {"runs": runs, "decode": []}

    def one(label: str, ids: list[int], ahead: bool, cold: bool = False) -> dict:
        os.environ[FLAG] = "1" if ahead else "0"
        extra = {"ahead": ahead}
        if cold:
            extra["drop"] = table.drop()
        return nr.run(engine, ids, f"{label} {'on' if ahead else 'off'}", table, runs, extra)

    syn8, syn32, syn128 = (pp.prompt_ids(model, n, 4242 + n) for n in (8192, 32768, 131072))
    nat = pp.prompt_ids(model, 32768, 515151, natural=True)
    pairs = {}
    for label, ids, order, cold in (("synthetic 8k", syn8, (False, True), False),
                                    ("synthetic 32k", syn32, (True, False), False),
                                    ("natural 32k warm", nat, (False, True), False),
                                    ("natural 32k cold", nat, (True, False), True),
                                    ("synthetic 128k", syn128, (False, True), False)):
        got = {ahead: one(label, ids, ahead, cold) for ahead in order}
        pairs[label] = {"on_tok_s": got[True]["tok_s"], "off_tok_s": got[False]["tok_s"],
                        "identical": got[True]["logits"] == got[False]["logits"],
                        "gain": round(got[True]["tok_s"] / got[False]["tok_s"] - 1, 4)}
    report["pairs"] = pairs
    chat = chat_ids(model, CHAT)
    for i in range(3):
        for ahead in (True, False):
            os.environ[FLAG] = "1" if ahead else "0"
            d = {"ahead": ahead, **decode(engine, chat, 512)}
            report["decode"].append(d)
            print(f"[ahead] decode {'on' if ahead else 'off'}: {d['tok_s']:.2f} tok/s, {d['tokens']} tokens, "
                  f"{d['rounds']} rounds, ids {d['ids_sha']}", flush=True)
    os.environ[FLAG] = "1"
    on = [d["tok_s"] for d in report["decode"] if d["ahead"]]
    off = [d["tok_s"] for d in report["decode"] if not d["ahead"]]
    report["decode_median"] = {"on": statistics.median(on), "off": statistics.median(off)}
    checks = {k: v["identical"] for k, v in pairs.items()}
    checks["decode ids"] = len({d["ids_sha"] for d in report["decode"]}) == 1
    report["checks"] = checks
    for label, v in pairs.items():
        print(f"[ahead] {label}: on {v['on_tok_s']:.0f} / off {v['off_tok_s']:.0f} tok/s ({100 * v['gain']:+.1f}%), "
              f"head logits {'identical' if v['identical'] else 'DIFFER'}", flush=True)
    print(f"[ahead] decode median: on {report['decode_median']['on']:.2f} / off {report['decode_median']['off']:.2f} "
          f"tok/s; ids {'identical' if checks['decode ids'] else 'DIFFER'}", flush=True)
    (out / "stage_ahead.json").write_text(json.dumps(report, indent=1, default=str))
    passed = sum(1 for v in checks.values() if v)
    (out / "ledger_metrics.json").write_text(json.dumps({
        "prefill_8k": pairs["synthetic 8k"]["on_tok_s"], "prefill_32k": pairs["synthetic 32k"]["on_tok_s"],
        "prefill_128k": pairs["synthetic 128k"]["on_tok_s"], "decode_1": report["decode_median"]["on"],
        "tests_passed": passed, "tests_failed": len(checks) - passed}))
    print(f"[ahead] checks {passed}/{len(checks)}; done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
