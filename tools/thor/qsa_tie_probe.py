"""Why a short prompt's reply changes with Thor's prompt QSA on (THOR-K4-QSA; the K4-QSA gate's short-haiku token 17).

    python tools/thor/qsa_tie_probe.py MODEL_DIR OUT_DIR [--count 32] [--diag 256]

One engine as served (int8 KV, --parallel 4, every Thor switch as this process sets it), the same requests in three
modes: ``first`` (Thor's prompt QSA on every prompt block, as first gated), ``dense_tf`` (blocks with no sparse row on
TensorFold's kernels, ``thor_qsa.DENSE_ON_TF``) and ``tf`` (Thor's prompt QSA off, TensorFold's prompt kernels). Kept
prompt states are dropped before every request, so each fills from row 0.

1. Attention bits on short-haiku's prompt pass in ``first``: each Thor attention call (layer, MTP head) against
   TensorFold's kernel on the same inputs (the same selections): elements that differ, rows touched, largest
   difference. No row of this prompt is sparse, so both read every key up to their own.
2. The prompt pass's head logits per mode (a hash, bit for bit).
3. Replies: thor_divergence.py's short probes (templated by the CUDA server's own App, as ``collect`` sends them),
   greedy, end tokens honoured, ``--count`` tokens at most, MTP drafts on (as served) and off; with ``--diag N``,
   also the twelve prompts of the decode diagnosis (N tokens, end tokens ignored, drafts on). Per mode: the tokens,
   the first index where they leave ``tf``'s, MTP drafted and accepted rows.
Writes qsa_tie_probe.json; prints one line per finding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ngram_residency as nr  # noqa: E402
import prefill_profile as pp  # noqa: E402
import thor_divergence as td  # noqa: E402

DIAG = (
    "Explain how a refrigerator works, step by step.",
    "Write a short story about a robot learning to paint.",
    "Describe the causes and effects of the French Revolution.",
    "What are the main differences between TCP and UDP? Give examples.",
    "Give a detailed recipe for a vegetable lasagna.",
    "Explain the theory of evolution to a high school student.",
    "Write a Python function that merges two sorted lists, and explain it.",
    "Summarize the plot of a classic adventure novel of your choice.",
    "What should someone consider when buying a used car?",
    "Describe how vaccines train the immune system.",
    "Write a letter to a friend about a recent trip to the mountains.",
    "Explain what a hash table is and how collisions are handled.",
)
MODES = ("first", "dense_tf", "tf")


def set_mode(thor_qsa, mode: str) -> None:
    thor_qsa.set_active(mode != "tf")
    thor_qsa.DENSE_ON_TF = mode == "dense_tf"


def reply(engine, ids: list[int], count: int, draft: bool, stop_eos: bool, bits: "AttnBits") -> dict:
    """One request through the concurrent decoder, as served; Thor's prompt-attention calls counted in the rounds
    after the prompt has filled (decode and MTP verify windows: none expected)."""

    from tensorfold.cuda.streams import Stream

    nr.forget(engine)
    multi = engine.multi
    s = Stream(list(ids), count, None, draft=draft, stop_eos=stop_eos)
    multi.admit(s)
    in_decode = 0
    while not s.done:
        decoding, before = bool(s.out), bits.calls
        multi.finish(multi.round())
        if decoding:
            in_decode += bits.calls - before
    if s.error is not None:
        raise s.error
    return {"tokens": list(s.out), "drafted": s.drafted, "accepted": s.accepted, "rounds": s.rounds,
            "thor_attention_calls_in_decode_rounds": in_decode}


def first_diff(a: list[int], b: list[int]) -> int | None:
    i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), None)
    return i if i is not None or len(a) == len(b) else min(len(a), len(b))


class AttnBits:
    """Thor's attention calls in ``first`` mode against TensorFold's kernel on the same inputs."""

    def __init__(self, forward, attn_mod, thor_qsa) -> None:
        self.on, self.layer, self.rows, self.calls = False, None, [], 0
        orig_block, orig_thor = forward.attn_block, thor_qsa.attention

        def attn_block(layer, w, segs, b, R, mtp, context=None):
            self.layer = "mtp" if mtp else int(layer.index)
            return orig_block(layer, w, segs, b, R, mtp, context=context)

        def attention(q, cache, shadow, pos0, scratch, rows, scale, out, *, context, bits):
            got = orig_thor(q, cache, shadow, pos0, scratch, rows, scale, out, context=context, bits=bits)
            self.calls += 1
            if self.on:
                want = torch.empty_like(out[:rows])
                attn_mod.attention(q, cache.k, cache.v, pos0, scratch, rows, scale, out=want, context=context,
                                   ks=cache.ks, vs=cache.vs, bits=bits)
                a, b = got[:rows].float(), want.float()
                diff = a != b
                self.rows.append({"layer": self.layer, "p0": int(pos0.reshape(-1)[0]), "rows": int(rows),
                                  "elements": int(diff.numel()), "differ": int(diff.sum()),
                                  "rows_touched": int(diff.reshape(rows, -1).any(1).sum()),
                                  "max_abs": float((a - b).abs().max()), "max_ref": float(b.abs().max())})
            return got

        forward.attn_block, thor_qsa.attention = attn_block, attention


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--count", type=int, default=32)
    ap.add_argument("--diag", type=int, default=0)
    args = ap.parse_args()
    from tensorfold.cuda.server import App
    from tensorfold.families.qwen4_exp.cuda import attention as attn_mod
    from tensorfold.families.qwen4_exp.cuda import forward, thor_qsa

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    engine = pp.build_engine(args.model, 4, "int8")
    if not thor_qsa.use():
        raise SystemExit("Thor's prompt QSA is not on in this process (not sm_110, TENSORFOLD_THOR_QSA=0, or bf16 KV)")
    app = App(engine, args.model, "qsa-tie-probe", context_window=engine.max_len)
    probes = {p["name"]: p for p in td.prompts()}
    shorts = [n for n in probes if n.startswith("short-")]
    ids = {n: list(app.prepare(td.body(probes[n]["prompt"], probes[n]["max_tokens"], 5), True).prompt) for n in shorts}
    report: dict = {"model": str(args.model), "count": args.count, "attention": [], "logits": {}, "replies": {}}
    bits = AttnBits(forward, attn_mod, thor_qsa)

    # 1. attention bits on short-haiku's prompt pass, Thor's kernels on its dense blocks
    set_mode(thor_qsa, "first")
    nr.forget(engine)
    bits.on = True
    pp.run_prompt(engine, ids["short-haiku"])
    bits.on = False
    report["attention"] = bits.rows
    main_rows = [r for r in bits.rows if r["layer"] != "mtp"]
    print(f"[tie] short-haiku ({len(ids['short-haiku'])} tokens), Thor's attention vs TensorFold's on the same inputs, "
          f"{len(main_rows)} main-model calls: {sum(r['differ'] for r in main_rows)} of "
          f"{sum(r['elements'] for r in main_rows)} elements differ, in {sum(r['rows_touched'] for r in main_rows)} "
          f"row-layers; largest {max((r['max_abs'] for r in main_rows), default=0):.3g} "
          f"(of values up to {max((r['max_ref'] for r in main_rows), default=0):.3g})", flush=True)
    for r in bits.rows:
        print(f"[tie]   layer {r['layer']}: {r['differ']}/{r['elements']} elements differ, {r['rows_touched']}/"
              f"{r['rows']} rows, max |diff| {r['max_abs']:.3g}", flush=True)

    # 2. the prompt pass's head logits per mode
    for name in shorts:
        report["logits"][name] = {}
        for mode in MODES:
            set_mode(thor_qsa, mode)
            nr.forget(engine)
            report["logits"][name][mode] = pp.run_prompt(engine, ids[name])["logits_sha"]
        h = report["logits"][name]
        print(f"[tie] {name}: prompt-pass head logits first {'==' if h['first'] == h['tf'] else '!='} tf, dense_tf "
              f"{'==' if h['dense_tf'] == h['tf'] else '!='} tf", flush=True)

    # 3. replies
    jobs = [(name, ids[name], args.count, draft, True) for name in shorts for draft in (True, False)]
    if args.diag:
        for i, text in enumerate(DIAG):
            body = {"messages": [{"role": "user", "content": text}], "max_tokens": args.diag,
                    "chat_template_kwargs": {"enable_thinking": False}}
            jobs.append((f"diag-{i}", list(app.prepare(body, True).prompt), args.diag, True, False))
    for name, prompt, count, draft, stop_eos in jobs:
        key = f"{name} drafts {'on' if draft else 'off'}"
        got = {}
        for mode in MODES:
            set_mode(thor_qsa, mode)
            got[mode] = reply(engine, prompt, count, draft, stop_eos, bits)
        ref = got["tf"]["tokens"]
        in_decode = sum(g["thor_attention_calls_in_decode_rounds"] for g in got.values())
        row = {m: {**g, "first_diff_vs_tf": first_diff(g["tokens"], ref),
                   "sha": hashlib.sha256(json.dumps(g["tokens"]).encode()).hexdigest()[:12]} for m, g in got.items()}
        report["replies"][key] = row
        print(f"[tie] {key}: first leaves tf at {row['first']['first_diff_vs_tf']}, dense_tf at "
              f"{row['dense_tf']['first_diff_vs_tf']} (of {len(ref)} tokens); accepted/drafted "
              + ", ".join(f"{m} {g['accepted']}/{g['drafted']}" for m, g in got.items())
              + f"; Thor attention calls in decode rounds: {in_decode}", flush=True)
        (args.out / "qsa_tie_probe.json").write_text(json.dumps(report, indent=1))
    set_mode(thor_qsa, "dense_tf")
    print(f"[tie] done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
