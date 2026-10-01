"""K4 A/B on Jetson Thor (THOR-K4): one engine, the same prompts on TensorFold's DeltaNet prompt chain (``stock``) and
on thor_gdn's chunked chain (``thor``).

    python tools/thor/k4_check.py MODEL_DIR OUT_DIR [--events 8192,32768,131072]

The engine is built as served (int8 KV, --parallel 4: the Thor experts, stage-ahead and thor_dense with its default
weights, bf16 copies since HD-2026-10-01-07; TENSORFOLD_THOR_DENSE_WEIGHTS=dequant gives r11's recipe) and thor_gdn is
switched per mode with ``thor_gdn.set_active``; nothing is kept between prompts (each fills from row 0). Per mode: kprof
events at each ``--events`` length (the GPU timeline by kernel class: the DeltaNet prompt class, gdn.chain/front/back,
and the rest), then plain wall time for synthetic 8k, 32k and 128k and natural-text 32k warm and with the n-gram table's
pages dropped, in two orders (stock then thor, then thor then stock); each run's head-logit hash (the same prompt in
both orders must hash the same on one path: determinism); and follow-ups of 16k, 32k and 48k prompts (40 new tokens,
resumed from the kept point), twice per mode in alternating order. Writes k4_check.json and ledger_metrics.json
(prefill_* of ``thor`` in the first order, as first requests; tests: the determinism checks).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ngram_residency as nr  # noqa: E402
import prefill_profile as pp  # noqa: E402

GDN_LABELS = ("gdn.chain", "gdn.front", "gdn.back")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--events", default="8192,32768,131072")
    args = ap.parse_args()
    from tensorfold.cuda.kernels import thor_gdn
    from tensorfold.families.qwen4_exp.cuda.decode import entry_end

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    engine = pp.build_engine(args.model, 4, "int8")
    if not thor_gdn.S.ready:
        raise SystemExit("thor_gdn is not prepared in this process (not sm_110, or TENSORFOLD_THOR_GDN=0)")
    w = engine.w
    pp.instrument(w, w.cfg)
    table = nr.Table(w)
    report: dict = {"dtype": str(thor_gdn.S.dtype), "runs": [], "events": {}}
    runs = report["runs"]
    syn = {n: pp.prompt_ids(args.model, n, 4242 + n) for n in (8192, 32768, 131072)}
    nat = pp.prompt_ids(args.model, 32768, 515151, natural=True)
    prompts = (("synthetic 8k", syn[8192], False), ("synthetic 32k", syn[32768], False),
               ("natural 32k warm", nat, False), ("natural 32k cold", nat, True),
               ("synthetic 128k", syn[131072], False))

    def events(mode: str, n: int) -> None:
        thor_gdn.set_active(mode == "thor")
        nr.forget(engine)
        pp.L.reset("events")
        got = pp.run_prompt(engine, syn.get(n) or pp.prompt_ids(args.model, n, 4242 + n))
        summary = pp.events_summary(pp.L.marks)
        paths = summary["paths"]
        gdn = {lab: sum(v for k, v in paths.items() if k.split("/")[-1] == lab) for lab in GDN_LABELS}
        cls = summary["classes"]
        report["events"][f"{mode} {n}"] = {"tok_s": got["tok_s"], "seconds": got["seconds"], "classes": cls,
                                           "gdn": gdn, "total_ms": summary["total_ms"], "logits": got["logits_sha"]}
        pp.L.marks = []
        pp.L.reset("plain")
        print(f"[k4] events {n} {mode}: {got['tok_s']:.0f} tok/s; DeltaNet prompt class "
              f"{cls.get('GDN prefill', 0):.0f} ms of {summary['total_ms']:.0f} "
              f"({100 * cls.get('GDN prefill', 0) / summary['total_ms']:.1f}%): chain {gdn['gdn.chain']:.0f}, front "
              f"{gdn['gdn.front']:.0f}, back {gdn['gdn.back']:.0f} ms", flush=True)

    def plain(mode: str, order: int) -> None:
        thor_gdn.set_active(mode == "thor")
        for label, ids, cold in prompts:
            extra = {"mode": mode, "order": order}
            if cold:
                extra["drop"] = table.drop()
            nr.run(engine, ids, f"{label} {mode} {order}", table, runs, extra)

    def followup(mode: str, n: int, seed: int) -> None:
        """An n-token prompt, then the same prompt and 40 new tokens: the second resumes from the first's kept point (on
        this path, below thor_gdn.LIMIT, the last 64-row boundary before its end: up to 63 more rows than stock)."""

        thor_gdn.set_active(mode == "thor")
        nr.forget(engine)
        base = pp.prompt_ids(args.model, n, seed)
        pp.run_prompt(engine, base)
        cached = entry_end(base)                       # the kept prefix the follow-up resumes from
        got = pp.run_prompt(engine, base + pp.prompt_ids(args.model, 512, seed + 1)[-40:])
        report["followup"].setdefault(f"{mode} {n}", []).append({"seconds": got["seconds"], "tokens": got["tokens"],
                                                                 "kept_tokens": cached, "logits": got["logits_sha"]})
        print(f"[k4] follow-up {mode} {n}: {got['tokens']} tokens resumed in {got['seconds']:.3f}s (kept prefix "
              f"{cached})", flush=True)

    lengths = [int(v) for v in args.events.split(",") if v]
    for n in lengths:
        for mode in ("stock", "thor"):
            events(mode, n)
    report["followup"] = {}
    for n in (16384, 32768, 49152):                      # below the chunked-row limit, at it, past it
        for i, mode in enumerate(("stock", "thor", "thor", "stock")):
            followup(mode, n, 777 + 10 * i + n)
    plain("stock", 1)
    plain("thor", 1)
    plain("thor", 2)
    plain("stock", 2)
    thor_gdn.set_active(True)
    by = {r["label"]: r for r in runs}
    table_rows, checks = {}, {}
    for label, _, _ in prompts:
        s1, t1, t2, s2 = (by[f"{label} {m}"] for m in ("stock 1", "thor 1", "thor 2", "stock 2"))
        row = {"stock": [s1["tok_s"], s2["tok_s"]], "thor": [t1["tok_s"], t2["tok_s"]],
               "gain_first": t1["tok_s"] / s1["tok_s"] - 1, "gain_second": t2["tok_s"] / s2["tok_s"] - 1,
               "first_token_as_stock": t1["first"] == s1["first"]}
        table_rows[label] = row
        checks[f"{label}: thor deterministic"] = t1["logits"] == t2["logits"]
        checks[f"{label}: stock deterministic"] = s1["logits"] == s2["logits"]
        print(f"[k4] {label}: stock {s1['tok_s']:.0f}/{s2['tok_s']:.0f}, thor {t1['tok_s']:.0f}/{t2['tok_s']:.0f} "
              f"tok/s "
              f"({100 * row['gain_first']:+.1f}% / {100 * row['gain_second']:+.1f}%); thor logits the same both times "
              f"{checks[f'{label}: thor deterministic']}; first token as stock {row['first_token_as_stock']}",
              flush=True)
    ev = report["events"]
    for n in lengths:
        s, t = ev[f"stock {n}"], ev[f"thor {n}"]
        sg, tg = s["classes"].get("GDN prefill", 0), t["classes"].get("GDN prefill", 0)
        report[f"in_situ_{n}"] = {"gdn_stock_ms": sg, "gdn_thor_ms": tg, "gdn_x": sg / tg if tg else None,
                                  "gdn_share_stock": sg / s["total_ms"], "gdn_share_thor": tg / t["total_ms"],
                                  "chain_x": s["gdn"]["gdn.chain"] / t["gdn"]["gdn.chain"] if t["gdn"]["gdn.chain"]
                                  else None, "tok_s": [s["tok_s"], t["tok_s"]]}
        r = report[f"in_situ_{n}"]
        print(f"[k4] in situ {n}: DeltaNet prompt class {sg:.0f} -> {tg:.0f} ms (x{r['gdn_x']:.2f}; share "
              f"{100 * r['gdn_share_stock']:.1f}% -> {100 * r['gdn_share_thor']:.1f}%), chain x{r['chain_x']:.2f}; "
              f"{s['tok_s']:.0f} -> {t['tok_s']:.0f} tok/s", flush=True)
    report["table"], report["checks"] = table_rows, checks
    (args.out / "k4_check.json").write_text(json.dumps(report, indent=1, default=str))
    passed = sum(1 for v in checks.values() if v)
    (args.out / "ledger_metrics.json").write_text(json.dumps({
        "prefill_8k": table_rows["synthetic 8k"]["thor"][0], "prefill_32k": table_rows["synthetic 32k"]["thor"][0],
        "prefill_128k": table_rows["synthetic 128k"]["thor"][0], "tests_passed": passed,
        "tests_failed": len(checks) - passed}))
    print(f"[k4] checks {passed}/{len(checks)}; done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
