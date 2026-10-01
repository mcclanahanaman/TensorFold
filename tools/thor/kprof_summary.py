"""Tables from ``prefill_profile.py``'s kprof.json: each prompt length's prefill wall time by kernel class.

    python tools/thor/kprof_summary.py OUT_DIR/kprof.json [--json SUMMARY.json]

Per class: kernel time from CUPTI (``profile``; at lengths where only some passes were profiled, each class's
kernel share of its CUDA-event time on those passes, times its event time over every pass), the CUDA-event time of
the class (``events``: kernel time plus any GPU idle inside its labels), achieved TFLOPS on the analytic FLOP counts,
and the share of the uninstrumented wall time (``plain``). "host gaps / idle" is the plain wall time not covered by
kernels.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ORDER = ("routed GEMM (FI W4A16)", "routed GEMM (portable, MTP layer)", "routed glue (route/gather/scatter)",
         "shared expert", "dense prompt GEMMs", "QSA / attention", "GDN prefill", "norms / elementwise",
         "host-side steps (stage, sample)", "unattributed")


def klass_of_label(label: str) -> str:
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from kprof_classes import klass

    return klass(tuple(label.split("/")))


def summarize(report: dict) -> dict:
    runs = [r for r in report["runs"] if not r.get("text") and not r.get("rows")]   # the main runs (no sweeps)
    lengths = sorted({r["target"] for r in runs})
    out = {}
    for n in lengths:
        plain = next((r for r in runs if r["target"] == n and r["mode"] == "plain"), None)
        ev = next((r for r in runs if r["target"] == n and r["mode"] == "events"), None)
        pr = next((r for r in runs if r["target"] == n and r["mode"] == "profile"), None)
        wall = plain["seconds"] * 1000.0 if plain else (ev["seconds"] * 1000.0 if ev else None)
        row: dict = {"tokens": (plain or ev or pr)["tokens"], "plain_ms": plain and plain["seconds"] * 1000.0,
                     "plain_tok_s": plain and plain["tok_s"], "events_ms": ev and ev["seconds"] * 1000.0,
                     "profile_ms": pr and pr["seconds"] * 1000.0}
        ev_cls = defaultdict(float, (ev or {}).get("events", {}).get("classes", {}))
        busy = defaultdict(float)
        if pr and pr.get("profile"):
            prof = pr["profile"]
            passes = len((ev or {}).get("events", {}).get("passes", {}))
            per_pass_ev = (ev or {}).get("events", {}).get("passes", {})
            if passes and len(prof) < passes - 1 and ev:          # sampled passes: scale by the events run
                num, den = defaultdict(float), defaultdict(float)
                for i, s in prof.items():
                    for c, v in s["classes"].items():
                        num[c] += v
                    for c, v in per_pass_ev.get(i, {}).items():
                        den[c] += v
                for c in set(num) | set(ev_cls):
                    r = num[c] / den[c] if den.get(c) else 1.0
                    busy[c] = min(r, 1.0) * ev_cls.get(c, 0.0) if c in ev_cls else num[c] * passes / len(prof)
                row["profile_scaled"] = True
            else:
                for s in prof.values():
                    for c, v in s["classes"].items():
                        busy[c] += v
                row["profile_scaled"] = False
            row["profiled_passes"] = len(prof)
            row["trace_busy_ms"] = sum(s["busy_ms"] for s in prof.values())
            row["trace_span_ms"] = sum(s["span_ms"] for s in prof.values())
            row["unattributed_ms"] = sum(s["unattributed_ms"] for s in prof.values())
        flops = defaultdict(float)
        nbytes = defaultdict(float)
        for label, f in (ev or pr or {}).get("flops", {}).items():
            flops[klass_of_label(label)] += f
        for label, b in (ev or pr or {}).get("bytes", {}).items():
            nbytes[klass_of_label(label)] += b
        classes = {}
        total_busy = 0.0
        for c in ORDER:
            b = busy.get(c, 0.0)
            total_busy += b
            classes[c] = {"busy_ms": b, "events_ms": ev_cls.get(c, 0.0),
                          "pct_wall": 100.0 * b / wall if wall else None,
                          "tflops": flops[c] / (b * 1e9) if b and flops.get(c) else None,
                          "gb_s": nbytes[c] / (b * 1e6) if b and nbytes.get(c) else None}
        idle = wall - total_busy if wall else None
        classes["host gaps / idle"] = {"busy_ms": idle, "events_ms": ev_cls.get("host gaps / idle", 0.0),
                                       "pct_wall": 100.0 * idle / wall if wall else None}
        row["classes"] = classes
        row["busy_total_ms"] = total_busy
        # labels in detail (events time and, when profiled at every pass, kernel time)
        detail = defaultdict(lambda: {"events_ms": 0.0, "busy_ms": 0.0, "flops": 0.0, "calls": 0})
        for path, v in (ev or {}).get("events", {}).get("paths", {}).items():
            detail[path.split("/")[-1]]["events_ms"] += v
        if pr and pr.get("profile") and not row.get("profile_scaled"):
            for s in pr["profile"].values():
                for path, v in s["paths"].items():
                    detail[path.split("/")[-1]]["busy_ms"] += v
        for label, f in (ev or {}).get("flops", {}).items():
            detail[label]["flops"] += f
        for label, c in (ev or {}).get("calls", {}).items():
            detail[label]["calls"] += c
        row["labels"] = dict(sorted(detail.items(), key=lambda kv: -kv[1]["events_ms"]))
        out[n] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    report = json.loads(args.report.read_text())
    s = summarize(report)
    for n, row in s.items():
        print(f"\n== {row['tokens']} tokens: plain {row['plain_ms'] / 1000:.2f} s = {row['plain_tok_s']:.0f} tok/s; "
              f"events run {row['events_ms'] / 1000 if row['events_ms'] else float('nan'):.2f} s; "
              f"profiled passes {row.get('profiled_passes')} (scaled: {row.get('profile_scaled')}); "
              f"unattributed {row.get('unattributed_ms', 0):.1f} ms")
        print(f"{'class':40s} {'kernel ms':>10s} {'% wall':>7s} {'events ms':>10s} {'TFLOPS':>7s} {'GB/s':>7s}")
        for c, v in row["classes"].items():
            tf = f"{v['tflops']:.1f}" if v.get("tflops") else ""
            gb = f"{v['gb_s']:.0f}" if v.get("gb_s") else ""
            print(f"{c:40s} {v['busy_ms']:10.1f} {v['pct_wall']:7.1f} {v['events_ms']:10.1f} {tf:>7s} {gb:>7s}")
        print("  top labels (events ms / kernel ms / TFLOPS / calls):")
        for label, v in list(row["labels"].items())[:28]:
            t = v["busy_ms"] or v["events_ms"]
            tf = f"{v['flops'] / (t * 1e9):.1f}" if v["flops"] and t else ""
            print(f"    {label:28s} {v['events_ms']:9.1f} {v['busy_ms']:9.1f} {tf:>7s} {v['calls']:6d}")
    if args.json:
        args.json.write_text(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
