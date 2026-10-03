"""tools/thor/hc_check_summary.py: hc_check.py kernels' per-call rows -> per prompt and row count M, over layers.

thor_hc2's columns (THOR-HC-FUSE-2) go in a second table when present.

Usage: python tools/thor/hc_check_summary.py HC_CHECK.json [--json FILE] [--markdown FILE]
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

HEADER = ("| prompt | M | calls | act equal | inject equal | mixed equal "
          "| act err ratio max | mixed err ratio max | r25 ms | new ms | speedup |")


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _ratio(r25, new):
    """new/r25, with 0/0 counted as 1.0 and x/0 with x > 0 as math.inf."""
    if r25 == 0:
        return 1.0 if new == 0 else math.inf
    return new / r25


def summarize(rows: list) -> dict:
    """Group the per-call rows by prompt then str(M), both in first-seen order; reduce each group."""
    groups = {}
    for row in rows:
        groups.setdefault((row["prompt"], str(row["M"])), []).append(row)
    out = {}
    for (prompt, m), group in groups.items():
        stats = {"calls": len(group)}
        errors = [row[k] for row in group for k in row if k.endswith("_error")]
        if errors:
            stats["errors"] = errors
        for key in ("equal_act", "equal_inject", "equal_mixed", "equal_act2", "equal_mixed2"):
            usable = [row[key] for row in group if key in row and row[key] is not None]
            stats[key] = all(usable) if usable else None
        for x in ("act", "mixed"):
            ratios = [_ratio(row["err_%s_r25" % x], row["err_%s_new" % x]) for row in group
                      if _num(row.get("err_%s_r25" % x)) and _num(row.get("err_%s_new" % x))]
            if ratios:
                stats["err_ratio_%s_max" % x] = max(ratios)
        for prefix, fun in (("errmax_", max), ("ms_", statistics.median)):
            keys = sorted({k for row in group for k in row
                           if k.startswith(prefix) and not k.endswith("_error")})
            for key in keys:
                usable = [row[key] for row in group if key in row and _num(row[key])]
                if usable:
                    stats[key] = fun(usable)
        if _num(stats.get("ms_r25")) and _num(stats.get("ms_new")) and stats["ms_new"] > 0:
            stats["speedup"] = stats["ms_r25"] / stats["ms_new"]
        if _num(stats.get("ms_new")) and _num(stats.get("ms_new2")) and stats["ms_new2"] > 0:
            stats["speedup2"] = stats["ms_new"] / stats["ms_new2"]
        out.setdefault(prompt, {})[m] = stats
    return out


def _flag(v):
    return "-" if v is None else ("yes" if v else "no")


def _cell(v, spec):
    return "-" if v is None else format(v, spec)


HEADER2 = ("| prompt | M | calls | act2 equal | mixed2 equal | new ms | new2 ms | new2 down ms "
           "| new2 upmix ms | speedup2 |")


def markdown2(summary: dict) -> str:
    """Render the thor_hc2 columns as a second table, one row per (prompt, M) with second-path data."""
    lines = [HEADER2, "|" + "---|" * 10]
    found = False
    for prompt, rows in summary.items():
        for m, stats in rows.items():
            act2 = stats.get("equal_act2")
            if act2 is None and not _num(stats.get("ms_new2")):
                continue
            found = True
            cells = [str(prompt), str(m), str(stats.get("calls", 0)), _flag(act2),
                     _flag(stats.get("equal_mixed2"))]
            cells += [_cell(stats.get(key), ".3f")
                      for key in ("ms_new", "ms_new2", "ms_new2_down_act", "ms_new2_upmix")]
            cells.append(_cell(stats.get("speedup2"), ".2f"))
            lines.append("| " + " | ".join(cells) + " |")
    if not found:
        return ""
    return "\n".join(lines) + "\n"


def markdown(summary: dict) -> str:
    """Render the summary as a markdown table, one row per (prompt, M) in summary order."""
    lines = [HEADER, "|" + "---|" * 11]
    for prompt, rows in summary.items():
        for m, stats in rows.items():
            cells = [str(prompt), str(m), str(stats.get("calls", 0))]
            cells += [_flag(stats.get(key)) for key in ("equal_act", "equal_inject", "equal_mixed")]
            cells += [_cell(stats.get("err_ratio_act_max"), ".2f"),
                      _cell(stats.get("err_ratio_mixed_max"), ".2f")]
            cells += [_cell(stats.get("ms_r25"), ".3f"), _cell(stats.get("ms_new"), ".3f")]
            cells.append(_cell(stats.get("speedup"), ".2f"))
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Summarize hc_check.py per-call rows per prompt and M.")
    parser.add_argument("src", help="hc_check.py JSON: {'upmix': bool, 'rows': [...]} or a bare list of rows")
    parser.add_argument("--json", dest="json_out", metavar="FILE", help="also write {'upmix', 'summary'} to FILE")
    parser.add_argument("--markdown", dest="markdown_out", metavar="FILE",
                        help="also write the markdown table to FILE")
    args = parser.parse_args(argv)
    data = json.loads(Path(args.src).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        rows = data.get("rows", [])
        upmix = data.get("upmix")
        fuse2 = data.get("fuse2")
    else:
        rows, upmix, fuse2 = data, None, None
    summary = summarize(rows)
    if args.json_out:
        payload = json.dumps({"upmix": upmix, "fuse2": fuse2, "summary": summary}, indent=1) + "\n"
        Path(args.json_out).write_text(payload, encoding="utf-8")
    text = markdown(summary)
    text2 = markdown2(summary)
    if text2:
        text += "\n" + text2
    if args.markdown_out:
        Path(args.markdown_out).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
