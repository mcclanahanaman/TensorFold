"""K2 A/B on Jetson Thor (THOR-K2): one engine, the same prompts on TensorFold's prompt GEMMs, thor_dense with bf16
copies, and thor_dense dequantizing at each call.

    python tools/thor/k2_check.py MODEL_DIR OUT_DIR          (copies are thor_dense's default: HD-2026-10-01-07)

The engine is built as served (int8 KV, --parallel 4) with the copies made at load. Phase 1: per prompt, ``stock``
(``thor_dense.set_active(False)``) then ``copy``; phase 2 (the copies dropped): ``dequant``, and ``stock`` again on the
32k prompt. Each mode: kprof events at 32k (class times), plain wall time for synthetic 8k/32k/128k and natural-text
32k warm and with the n-gram table's pages dropped, and the head logits' hash (copy and dequant must agree). Writes
k2_check.json and ledger_metrics.json (prefill_* of ``copy``, the default since HD-2026-10-01-07).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ngram_residency as nr  # noqa: E402
import prefill_profile as pp  # noqa: E402


def main() -> None:
    from tensorfold.cuda.nvfp4 import thor_dense as td

    model, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    engine = pp.build_engine(model, 4, "int8")
    if not td.S.active:
        raise SystemExit("thor_dense is off in this process")
    w = engine.w
    pp.instrument(w, w.cfg)
    table = nr.Table(w)
    report: dict = {"runs": [], "events": {}}
    runs = report["runs"]
    held = sum(getattr(lin, "thor_w").numel() * 2 for _, lin, _ in td.faces(w) if getattr(lin, "thor_w", None)
               is not None)
    report["copy_bytes"] = held
    print(f"[k2] copies held: {held / 2**30:.2f} GiB", flush=True)
    syn8, syn32, syn128 = (pp.prompt_ids(model, n, 4242 + n) for n in (8192, 32768, 131072))
    nat = pp.prompt_ids(model, 32768, 515151, natural=True)
    prompts = (("synthetic 8k", syn8, False), ("synthetic 32k", syn32, False), ("natural 32k warm", nat, False),
               ("natural 32k cold", nat, True), ("synthetic 128k", syn128, False))

    def plain(mode: str) -> None:
        td.set_active(mode != "stock")
        for label, ids, cold in prompts:
            extra = {"mode": mode}
            if cold:
                extra["drop"] = table.drop()
            nr.run(engine, ids, f"{label} {mode}", table, runs, extra)

    def events(mode: str) -> None:
        td.set_active(mode != "stock")
        nr.forget(engine)
        pp.L.reset("events")
        got = pp.run_prompt(engine, syn32)
        summary = pp.events_summary(pp.L.marks)
        dense = {k: v for k, v in summary["paths"].items() if k.split("/")[-1].startswith("dense.")}
        report["events"][mode] = {"tok_s": got["tok_s"], "classes": summary["classes"], "dense_paths": dense,
                                  "shared_gemm_ms": sum(v for k, v in summary["paths"].items()
                                                        if k.endswith("moe.shared.gemm")),
                                  "logits": got["logits_sha"]}
        pp.L.marks = []
        pp.L.reset("plain")
        cls = summary["classes"]
        print(f"[k2] events 32k {mode}: {got['tok_s']:.0f} tok/s; dense {cls.get('dense prompt GEMMs', 0):.0f} ms, "
              f"shared expert {cls.get('shared expert', 0):.0f} ms, shared GEMMs "
              f"{report['events'][mode]['shared_gemm_ms']:.0f} ms", flush=True)

    for mode in ("stock", "copy"):
        events(mode)
    for mode in ("stock", "copy"):
        plain(mode)
    report["freed_bytes"] = td.release(w)
    events("dequant")
    plain("dequant")
    td.set_active(False)
    nr.run(engine, syn32, "synthetic 32k stock again", table, runs, {"mode": "stock"})
    td.set_active(True)
    by = {r["label"]: r for r in runs}
    checks, table_rows = {}, {}
    for label, _, _ in prompts:
        s, c, d = (by.get(f"{label} {m}") for m in ("stock", "copy", "dequant"))
        table_rows[label] = {"stock": s["tok_s"], "copy": c["tok_s"], "dequant": d["tok_s"],
                             "copy_gain": c["tok_s"] / s["tok_s"] - 1, "dequant_gain": d["tok_s"] / s["tok_s"] - 1,
                             "first_same_as_stock": c["first"] == s["first"]}
        checks[f"{label}: copy == dequant"] = c["logits"] == d["logits"]
        row = table_rows[label]
        print(f"[k2] {label}: stock {s['tok_s']:.0f} / copy {c['tok_s']:.0f} / dequant {d['tok_s']:.0f} tok/s "
              f"(copy {100 * row['copy_gain']:+.1f}%, dequant {100 * row['dequant_gain']:+.1f}%); copy == dequant "
              f"logits {checks[f'{label}: copy == dequant']}; first token as stock {row['first_same_as_stock']}",
              flush=True)
    ev = report["events"]
    for mode in ("copy", "dequant"):
        s_d = ev["stock"]["classes"].get("dense prompt GEMMs", 0)
        m_d = ev[mode]["classes"].get("dense prompt GEMMs", 0)
        s_all = s_d + ev["stock"]["shared_gemm_ms"]
        m_all = m_d + ev[mode]["shared_gemm_ms"]
        report[f"in_situ_{mode}"] = {"dense_x": s_d / m_d if m_d else None, "dense_and_shared_x": s_all / m_all}
        print(f"[k2] in situ at 32k, {mode}: dense prompt GEMMs {s_d:.0f} -> {m_d:.0f} ms (x{s_d / m_d:.2f}); with the "
              f"shared expert's GEMMs {s_all:.0f} -> {m_all:.0f} ms (x{s_all / m_all:.2f})", flush=True)
    checks["events copy == dequant"] = ev["copy"]["logits"] == ev["dequant"]["logits"]
    report["table"], report["checks"] = table_rows, checks
    (out / "k2_check.json").write_text(json.dumps(report, indent=1, default=str))
    passed = sum(1 for v in checks.values() if v)
    (out / "ledger_metrics.json").write_text(json.dumps({
        "prefill_8k": table_rows["synthetic 8k"]["copy"], "prefill_32k": table_rows["synthetic 32k"]["copy"],
        "prefill_128k": table_rows["synthetic 128k"]["copy"], "tests_passed": passed,
        "tests_failed": len(checks) - passed}))
    print(f"[k2] checks {passed}/{len(checks)}; done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
