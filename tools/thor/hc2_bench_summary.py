"""tools/thor/hc2_bench_summary.py (THOR-HC-FUSE-2): per-kernel timings of the hyper-connection read-out
kernels, a baseline path against a new one, with bytes and GB/s derived from a fixed bytes model.

Usage: python tools/thor/hc2_bench_summary.py BENCH.json [--base r31] [--new ws] [--json FILE] [--markdown FILE]
"""

import argparse
import json
import sys
from pathlib import Path


def bytes_model(kernel, M, S=4, D=2560, LOW=320, NC=10) -> int:
    """Bytes touched exactly once per tensor for one launch of `kernel` at batch M."""
    if kernel == "down":
        return 2 * M * S * D + 2 * (LOW + S) * S * D + 2 * M * (LOW + S) + 4 * M * NC * S
    if kernel == "upmix":
        return 2 * M * S * D + 2 * M * LOW + 2 * S * D * LOW + 2 * M * D + 4 * M * NC * S
    raise ValueError(f"unknown kernel: {kernel!r}")


def summarize(bench, base="r31", new="ws") -> list:
    """Best ms per (kernel, M) and path, with derived bytes, speedup, GB/s and equality."""
    groups, order = {}, []
    for row in bench.get("rows", []):
        if not isinstance(row.get("ms"), (int, float)):      # a failed measurement (an "error" row)
            continue
        if row["kernel"] not in order:
            order.append(row["kernel"])
        groups.setdefault((row["kernel"], row["M"]), []).append(row)
    out = []
    for kernel, m in sorted(groups, key=lambda key: (order.index(key[0]), key[1])):
        picked = {}
        for path in (base, new):
            mine = [row for row in groups[(kernel, m)] if row["path"] == path]
            if mine:
                picked[path] = (min(row["ms"] for row in mine), all(row["equal"] for row in mine))
        ms_base, eq_base = picked.get(base, (None, None))
        ms_new, eq_new = picked.get(new, (None, None))
        total = bytes_model(kernel, m)
        out.append({"kernel": kernel, "M": m, "bytes": total, "ms_base": ms_base, "ms_new": ms_new,
                    "speedup": round(ms_base / ms_new, 3) if ms_base is not None and ms_new is not None else None,
                    "gbs_base": round(total / (ms_base * 1e6), 1) if ms_base is not None else None,
                    "gbs_new": round(total / (ms_new * 1e6), 1) if ms_new is not None else None,
                    "equal_base": eq_base, "equal_new": eq_new})
    return out


def _cell(value, spec):
    """A number formatted per `spec`, or '-' when the path is missing."""
    return "-" if value is None else format(value, spec)


def markdown(rows, base="r31", new="ws") -> str:
    """Render the summarized rows as a markdown table; '-' marks a missing path."""
    lines = [f"| kernel | M | {base} ms | {new} ms | speedup | {base} GB/s | {new} GB/s | equal |",
             "|" + "---|" * 8]
    for row in rows:
        eq = "-" if row["equal_new"] is None else ("yes" if row["equal_new"] else "no")
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            row["kernel"], row["M"], _cell(row["ms_base"], ".3f"), _cell(row["ms_new"], ".3f"),
            _cell(row["speedup"], ".3f"), _cell(row["gbs_base"], ".1f"), _cell(row["gbs_new"], ".1f"), eq))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Summarize hc2 bench rows: best ms per kernel and M.")
    parser.add_argument("bench", help="bench JSON: {'meta', 'rows': [{'kernel', 'path', 'M', 'ms', 'equal'}]}")
    parser.add_argument("--base", default="r31", help="baseline path name (default r31)")
    parser.add_argument("--new", default="ws", help="new path name (default ws)")
    parser.add_argument("--json", dest="json_out", metavar="FILE", help="also write the summarized rows to FILE")
    parser.add_argument("--markdown", dest="markdown_out", metavar="FILE", help="also write the markdown to FILE")
    args = parser.parse_args(argv)
    bench = json.loads(Path(args.bench).read_text(encoding="utf-8"))
    rows = summarize(bench, base=args.base, new=args.new)
    text = markdown(rows, base=args.base, new=args.new)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"meta": bench["meta"], "rows": rows}, indent=1) + "\n",
                                       encoding="utf-8")
    if args.markdown_out:
        Path(args.markdown_out).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
