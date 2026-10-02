"""tools/thor/qsa_kernels_summary.py: qsa_check.py kernels' per-slice results -> per prompt and first row, over layers.

Usage: python tools/thor/qsa_kernels_summary.py KERNELS_JSON [--json OUT]
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

SKIP = ("prompt", "layer", "p0", "rows")
DEFAULT_PAIRS = [
    ("attn_thor_ms", "attn_new_ms"),
    ("attn_tf_ms", "attn_new_ms"),
    ("select_tf_ms", "select_new_ms"),
    ("rows_thor_ms", "rows_new_ms"),
    ("rows_tf_ms", "rows_new_ms"),
    ("select_tf_ms", "select_sel_ms"),
    ("select_new_ms", "select_sel_ms"),
]
FIELDS = [
    ("attn_tf_ms", 3),
    ("attn_thor_ms", 3),
    ("attn_new_ms", 3),
    ("select_tf_ms", 3),
    ("select_new_ms", 3),
    ("rows_thor_ms", 3),
    ("rows_new_ms", 3),
]
HEADER = ("| prompt | p0 | layers | attn tf ms | attn r19 ms | attn new ms | select tf ms | select new ms "
          "| rows r19 ms | rows new ms | attn r19/new | select tf/new | err new/tf max | select equal "
          "| select sel ms | select tf/sel | sel equal |")
SEPARATOR = "|---" * 17


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def summarize(results):
    """Group the per-layer dicts by prompt and str(p0) (first-seen order); reduce each metric over layers."""
    groups = {}
    for d in results:
        groups.setdefault((d["prompt"], str(d["p0"])), []).append(d)
    out = {}
    for (prompt, p0), group in groups.items():
        row = {"layers": len(group)}
        errors = set()
        for d in group:
            for key in d:
                if key in SKIP:
                    continue
                if key.endswith("_error"):
                    errors.add(key)
                    continue
                vals = [x[key] for x in group if key in x]
                if "_equal" in key:
                    usable = [v for v in vals if v is not None]
                    if usable:
                        row[key] = all(usable)
                elif key.startswith("err_ratio"):
                    usable = [v for v in vals if _num(v)]
                    if usable:
                        row[key] = max(usable)
                elif key.startswith("err_"):
                    usable = [v for v in vals if _num(v)]
                    if usable:
                        row[key] = sum(usable) / len(usable)
                elif key.endswith("_ms") or "_ms_" in key:
                    usable = [v for v in vals if _num(v)]
                    if usable:
                        row[key] = statistics.median(usable)
        if errors:
            row["errors"] = sorted(errors)
        out.setdefault(prompt, {})[p0] = row
    return out


def speedups(summary, pairs):
    """For every prompt/p0 of the summary, map each (A, B) pair present with numeric values and B != 0 to A/B."""
    out = {}
    for prompt, rows in summary.items():
        table = {}
        for p0, row in rows.items():
            entry = {}
            for a, b in pairs:
                if a in row and b in row and _num(row[a]) and _num(row[b]) and row[b] != 0:
                    entry[a + "/" + b] = row[a] / row[b]
            table[p0] = entry
        out[prompt] = table
    return out


def _cell(v, nd):
    if isinstance(v, bool):
        return "yes" if v else "no"
    if v is None:
        return "-"
    return f"{v:.{nd}f}" if nd else str(v)


def markdown(summary, sp):
    lines = [HEADER, SEPARATOR]
    for prompt, rows in summary.items():
        for p0, row in rows.items():
            speed = sp.get(prompt, {}).get(p0, {})
            cells = [str(prompt), str(p0), str(row.get("layers", 0))]
            cells += [_cell(row.get(key), nd) for key, nd in FIELDS]
            cells.append(_cell(speed.get("attn_thor_ms/attn_new_ms"), 2))
            cells.append(_cell(speed.get("select_tf_ms/select_new_ms"), 2))
            cells.append(_cell(row.get("err_ratio_new_max"), 2))
            cells.append(_cell(row.get("select_new_equal"), 0))
            cells.append(_cell(row.get("select_sel_ms"), 3))
            cells.append(_cell(speed.get("select_tf_ms/select_sel_ms"), 2))
            cells.append(_cell(row.get("select_sel_equal"), 0))
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize qsa_check.py kernels results per prompt and p0.")
    parser.add_argument("kernels_json", help="JSON list of per-layer kernel result dicts")
    parser.add_argument("--json", dest="json_out", metavar="OUT", help="also write summary and speedups to OUT")
    args = parser.parse_args(argv)
    results = json.loads(Path(args.kernels_json).read_text(encoding="utf-8"))
    summary = summarize(results)
    sp = speedups(summary, DEFAULT_PAIRS)
    print(markdown(summary, sp))
    if args.json_out:
        payload = json.dumps({"summary": summary, "speedups": sp}, indent=1) + "\n"
        Path(args.json_out).write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
