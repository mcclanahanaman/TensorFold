"""Digest of the ``[thor-graph]`` lines that a Thor decode-graph server logs (THOR-K3).

Reads one or more Thor server logs (utf-8, errors='replace') and keeps the lines that start with the prefix
``[thor-graph] `` followed by one JSON object (``on``, ``capture``, ``stats`` or ``fail`` events).  Writes a
markdown summary table (one row per log) to stdout and, with ``--json OUT``, a ``{'logs': [summary, ...]}``
document.

Usage:
    python tools/thor/graph_stats.py LOG [LOG ...] [--json OUT]        # prints the markdown table
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PREFIX = "[thor-graph] "
HEADER = ("| log | graphs on | verify hit | mtp hit | captures | capture s | graphs | evictions "
          "| reserved MiB | free drop MiB | failures |")
SEP = "|---|---|---|---|---|---|---|---|---|---|---|"


def parse(text: str) -> tuple[list[dict], int]:
    """Return (events, bad): the ``[thor-graph]`` JSON events in order, and the count of bad prefixed lines."""
    events, bad = [], 0
    for line in text.splitlines():
        if not line.startswith(PREFIX):
            continue
        try:
            obj = json.loads(line[len(PREFIX):])
        except ValueError:
            bad += 1
            continue
        if isinstance(obj, dict) and "event" in obj:
            events.append(obj)
        else:  # prefixed, but not one JSON object with an 'event' key
            bad += 1
    return events, bad


def summarize(events, label: str) -> dict:
    """Reduce the parsed events to one summary row for ``label``."""
    on, caps, stats, fail_kinds = None, [], None, []
    captures_by, max_capture_s = {}, None
    for e in events:
        kind = e.get("event")
        if kind == "on":
            if on is None:
                on = e
        elif kind == "capture":
            caps.append(e)
            key = f"{e.get('kind')}/{e.get('n')}"
            captures_by[key] = captures_by.get(key, 0) + 1
            cs = e.get("capture_s")
            if cs is not None and (max_capture_s is None or cs > max_capture_s):
                max_capture_s = cs
        elif kind == "stats":
            stats = e
        elif kind == "fail":
            fail_kinds.append(e.get("kind"))

    if stats is not None:
        forwards, replays = stats.get("forwards", {}), stats.get("replays", {})
        captures = stats.get("captures", {})
        hit_rate = {k: (replays.get(k, 0) / v if v else None) for k, v in forwards.items()}
        evictions, capture_s = stats.get("evictions"), stats.get("capture_s")
    else:
        forwards, replays, hit_rate, evictions = {}, {}, {}, None
        captures = {}
        for e in caps:
            captures[e.get("kind")] = captures.get(e.get("kind"), 0) + 1
        capture_s = sum(e.get("capture_s", 0) for e in caps) if caps else None

    if stats is not None:
        graphs, reserved_mib = stats.get("graphs", 0), stats.get("reserved_mib")
    else:
        graphs = caps[-1].get("graphs", 0) if caps else 0
        reserved_mib = caps[-1].get("reserved_mib") if caps else None

    src = stats if stats is not None else (caps[-1] if caps else None)
    if on is not None and src is not None and "free_mib" in on and "free_mib" in src:
        free_drop_mib = on["free_mib"] - src["free_mib"]
    else:
        free_drop_mib = None

    return {
        "label": label,
        "enabled": bool(events),
        "forwards": forwards,
        "replays": replays,
        "captures": captures,
        "hit_rate": hit_rate,
        "evictions": evictions,
        "capture_s": capture_s,
        "graphs": graphs,
        "reserved_mib": reserved_mib,
        "free_drop_mib": free_drop_mib,
        "failures": len(fail_kinds),
        "fail_kinds": fail_kinds,
        "captures_by": captures_by,
        "max_capture_s": max_capture_s,
    }


def _pct(v) -> str:
    return "-" if v is None else f"{100 * v:.1f}%"


def _mib(v) -> str:
    return "-" if v is None else f"{v:,.1f}"


def markdown(rows) -> str:
    """Render the summary rows as the markdown table (ends with a newline)."""
    lines = [HEADER, SEP]
    for r in rows:
        hit = r.get("hit_rate", {})
        cells = [
            r.get("label"),
            "yes" if r.get("enabled") else "no",
            _pct(hit.get("verify")),
            _pct(hit.get("mtp")),
            sum(r.get("captures", {}).values()),
            "-" if r.get("capture_s") is None else f"{r['capture_s']:.2f}",
            r.get("graphs", 0),
            "-" if r.get("evictions") is None else r["evictions"],
            _mib(r.get("reserved_mib")),
            _mib(r.get("free_drop_mib")),
            r.get("failures", 0),
        ]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Summarize the [thor-graph] lines of Thor decode-graph server logs (THOR-K3).")
    ap.add_argument("log", nargs="+", metavar="LOG", help="log file to scan")
    ap.add_argument("--json", metavar="OUT", default=None, help="write {'logs': [summary, ...]} to OUT")
    args = ap.parse_args(argv)

    rows = []
    for path in args.log:
        events, bad = parse(Path(path).read_text(encoding="utf-8", errors="replace"))
        summary = summarize(events, Path(path).stem)
        summary["bad_lines"] = bad
        rows.append(summary)

    print(markdown(rows), end="")

    if args.json:
        Path(args.json).write_text(json.dumps({"logs": rows}, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
