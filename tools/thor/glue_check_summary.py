"""tools/thor/glue_check_summary.py (THOR-EXPERT-GLUE): glue_check.json -> byte-equality counts, fp64 error ratios
and per-call timings.

Usage: python tools/thor/glue_check_summary.py CHECK.json [--json FILE] [--markdown FILE]
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

FLAGS = ("equal_h", "equal_pss", "equal_act", "equal_shared", "equal_served")
TIMINGS = ("ms_moe_off", "ms_moe_on", "ms_wb_off", "ms_wb_on", "ms_shared_off", "ms_shared_on", "ms_scatter")
HEADER = "| M | rows | all equal | err h off max | err h on max | err act off max | err act on max |"
ERR_MAXES = ("err_h_off_max", "err_h_on_max", "err_act_off_max", "err_act_on_max")


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _flag_counts(rows, flag):
    """(true count, count) over rows where the flag is a bool; None and absent are not counted."""
    usable = [row[flag] for row in rows if isinstance(row.get(flag), bool)]
    return sum(1 for v in usable if v), len(usable)


def summarize(check: dict) -> dict:
    """Reduce glue_check rows: equality counts, fp64 error ratios, per-M maxima and timing medians."""
    all_rows = check.get("rows", [])
    rows = [row for row in all_rows if "bench_error" not in row]
    errors = len(all_rows) - len(rows)
    equal = {flag: list(_flag_counts(rows, flag)) for flag in FLAGS}
    ratios = []
    for row in rows:
        for off, on in (("err_h_off_max", "err_h_on_max"), ("err_act_off_max", "err_act_on_max")):
            if _num(row.get(off)) and _num(row.get(on)) and row[off] != 0:
                ratios.append(row[on] / row[off])
    groups = {}
    for row in rows:
        groups.setdefault(str(row["M"]), []).append(row)
    by_m = {}
    for m, group in groups.items():
        stats = {"rows": len(group),
                 "all_equal": all(t == c for t, c in (_flag_counts(group, flag) for flag in FLAGS))}
        for key in ERR_MAXES:
            vals = [row[key] for row in group if _num(row.get(key))]
            if vals:
                stats[key] = max(vals)
        by_m[m] = stats
    timing = {}
    for key in TIMINGS:
        usable = [row[key] for row in rows if _num(row.get(key))]
        if usable:
            timing[key] = statistics.median(usable)
    saved = [(row["ms_moe_off"] + row["ms_wb_off"]) - (row["ms_moe_on"] + row["ms_wb_on"]) for row in rows
             if all(_num(row.get(key)) for key in ("ms_moe_off", "ms_wb_off", "ms_moe_on", "ms_wb_on"))]
    return {"rows": len(rows), "errors": errors, "equal": equal,
            "all_equal": not errors and all(t == c for t, c in equal.values()),
            "err_ratio_max": max(ratios) if ratios else None,
            "by_M": {m: by_m[m] for m in sorted(by_m, key=int)}, "timing": timing,
            "saved_ms_per_call": statistics.median(saved) if saved else None}


def _cell(v, spec):
    return "-" if v is None else format(v, spec)


def markdown(summary: dict) -> str:
    """Render the summary as markdown: a totals line, per-M error maxima, timing medians and saved ms."""
    lines = ["rows %d | errors %d | all equal %s" % (summary["rows"], summary["errors"], summary["all_equal"])]
    lines += [HEADER, "|" + "---|" * 7]
    for m, stats in summary["by_M"].items():
        cells = [m, str(stats["rows"]), str(stats["all_equal"])]
        cells += [_cell(stats.get(key), ".3e") for key in ERR_MAXES]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["| timing | median ms |", "|" + "---|" * 2]
    for key in TIMINGS:
        if key in summary["timing"]:
            lines.append(f"| {key} | {summary['timing'][key]:.3f} |")
    if summary["saved_ms_per_call"] is not None:
        lines.append("saved ms a call: {:.3f}".format(summary["saved_ms_per_call"]))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Summarize glue_check.py rows: equality, error ratios and timings.")
    parser.add_argument("src", help="glue_check.py JSON: {'note', 'active', 'layers', 'rows'}")
    parser.add_argument("--json", dest="json_out", metavar="FILE", help="also write the summary JSON to FILE")
    parser.add_argument("--markdown", dest="markdown_out", metavar="FILE", help="also write the markdown to FILE")
    args = parser.parse_args(argv)
    summary = summarize(json.loads(Path(args.src).read_text(encoding="utf-8")))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    text = markdown(summary)
    if args.markdown_out:
        Path(args.markdown_out).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
