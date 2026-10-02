"""Summarize pytest -q log files (counts, failures by file, stack-vs-pristine comparison).

Reads pytest console log files: one or more stack logs, optionally compared against pristine logs. Writes a JSON
summary and/or a Markdown report to the requested paths (the Markdown is also printed to stdout):

    python tools/thor/testlog_summary.py LOG [LOG ...] [--pristine LOG [LOG ...]] [--json FILE] [--markdown FILE]

Standard library only (argparse, json, re, pathlib, sys); no TensorFold and no third-party imports.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

COUNT_KEYS = ("passed", "failed", "error", "skipped", "xfailed", "xpassed", "warnings", "deselected")
BY_FILE_KEYS = ("passed", "failed", "error", "skipped", "xfailed", "xpassed")
BY_FILE_STATUS = {"PASSED": "passed", "FAILED": "failed", "ERROR": "error", "XFAIL": "xfailed", "XPASS": "xpassed"}

SUMMARY_RE = re.compile(
    r"^(?:no tests ran|(?P<counts>\d+ \w+(?:, \d+ \w+)*)) in (?P<secs>\d+(?:\.\d+)?)s(?: \(.*\))?$"
)
NODE_RE = re.compile(r"^(PASSED|FAILED|ERROR|XFAIL|XPASS) (?P<rest>.*)$")
SKIPPED_RE = re.compile(r"^SKIPPED \[(?P<n>\d+)\] (?P<rest>.*)$")


def parse_log(text: str) -> dict:
    """Parse one pytest -q log into complete/counts/seconds/failed/by_file."""
    lines = text.splitlines()
    out = {"complete": False, "counts": None, "seconds": None, "failed": [], "by_file": {}}
    match = None
    for line in lines:
        found = SUMMARY_RE.match(line.strip("= "))
        if found:
            match = found  # keep the LAST matching line
    if match is not None:
        out["complete"] = True
        out["seconds"] = float(match.group("secs"))
        counts = dict.fromkeys(COUNT_KEYS, 0)
        if match.group("counts"):
            for item in match.group("counts").split(", "):
                num, _, word = item.partition(" ")
                if word in ("error", "errors"):
                    counts["error"] += int(num)
                elif word in ("warning", "warnings"):
                    counts["warnings"] += int(num)
                elif word in counts:
                    counts[word] += int(num)
        out["counts"] = counts
    section = lines
    for index, line in enumerate(lines):
        if "short test summary info" in line:
            section = lines[index + 1:]
            break
    failed, by_file = [], {}
    for line in section:
        found = NODE_RE.match(line)
        if found:
            key = BY_FILE_STATUS[found.group(1)]
            node = found.group("rest").split(" - ")[0].strip()
            if key in ("failed", "error") and node not in failed:
                failed.append(node)
            by_file.setdefault(node.split("::")[0], dict.fromkeys(BY_FILE_KEYS, 0))[key] += 1
            continue
        found = SKIPPED_RE.match(line)
        if found:
            fname = found.group("rest").split(":", 1)[0]
            by_file.setdefault(fname, dict.fromkeys(BY_FILE_KEYS, 0))["skipped"] += int(found.group("n"))
    out["failed"] = failed
    out["by_file"] = by_file
    return out


def _merge_by_file(entries: list) -> dict:
    merged = {}
    for entry in entries:
        for fname, slot in entry["by_file"].items():
            target = merged.setdefault(fname, dict.fromkeys(BY_FILE_KEYS, 0))
            for key in BY_FILE_KEYS:
                target[key] += slot[key]
    return dict(sorted(merged.items()))


def _markdown(result: dict) -> str:
    lines = ["| log | passed | failed | error | skipped | seconds |", "|---|---|---|---|---|---|"]
    total_seconds = 0.0
    for entry in result["logs"]:
        name = Path(entry["path"]).name
        if not entry["complete"]:
            lines.append(f"| {name} | incomplete | - | - | - | - |")
            continue
        counts, seconds = entry["counts"], entry["seconds"]
        total_seconds += seconds
        lines.append(
            f"| {name} | {counts['passed']} | {counts['failed']} | {counts['error']} | {counts['skipped']}"
            f" | {seconds:.1f} |"
        )
    totals = result["totals"]
    lines.append(
        f"| total | {totals['passed']} | {totals['failed']} | {totals['error']} | {totals['skipped']}"
        f" | {total_seconds:.1f} |"
    )
    lines += ["", "failures by file:"]
    pristine_failed = set(result["pristine"]["failed"]) if result["pristine"] is not None else set()
    for fname, slot in result["by_file"].items():
        failing = slot["failed"] + slot["error"]
        if failing <= 0:
            continue
        row = f"- {fname}: {failing} failing"
        if result["pristine"] is not None:
            also = sum(1 for fid in result["failed"] if fid.split("::")[0] == fname and fid in pristine_failed)
            row += f" ({also} also on pristine)"
        lines.append(row)
    if result["pristine"] is not None:
        only = result["diff"]["stack_only"]
        lines.append("stack only: " + (", ".join(only) if only else "none"))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Summarize pytest -q logs (stack vs pristine).")
    parser.add_argument("logs", nargs="+", metavar="LOG", help="stack run log files")
    parser.add_argument("--pristine", nargs="+", metavar="LOG", default=None, help="pristine run log files")
    parser.add_argument("--json", metavar="FILE", default=None, help="write the JSON summary here")
    parser.add_argument("--markdown", metavar="FILE", default=None, help="write the Markdown report here")
    args = parser.parse_args(argv)

    def load(paths):
        return [{"path": p, **parse_log(Path(p).read_text(errors="ignore"))} for p in paths]

    entries = load(args.logs)
    stack_failed = sorted({fid for entry in entries for fid in entry["failed"]})
    result = {
        "logs": entries,
        "totals": {key: sum(e["counts"][key] for e in entries if e["complete"]) for key in COUNT_KEYS},
        "by_file": _merge_by_file(entries),
        "failed": stack_failed,
        "pristine": None,
        "diff": None,
    }
    if args.pristine is not None:
        pristine_entries = load(args.pristine)
        pristine_failed = sorted({fid for entry in pristine_entries for fid in entry["failed"]})
        result["pristine"] = {"logs": pristine_entries, "failed": pristine_failed}
        result["diff"] = {
            "stack_only": sorted(set(stack_failed) - set(pristine_failed)),
            "both": sorted(set(stack_failed) & set(pristine_failed)),
            "pristine_only": sorted(set(pristine_failed) - set(stack_failed)),
        }
    if args.json is not None:
        Path(args.json).write_text(json.dumps(result, indent=1))
    report = _markdown(result)
    print(report)
    if args.markdown is not None:
        Path(args.markdown).write_text(report + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
