"""THOR-DEC-1 memory digest of a run's out folder.

Per server tag, the digest reports the memory gate's room, the server's footprint from MemAvailable
and the driver's free memory at each probe phase, grow and eviction counts by segment between
probes, and each tag's difference from the same round's base config.

Usage:
    python tools/thor/stress_digest.py OUT_DIR --tags off_1,gs_1 [--base-config off] [--json OUT] [--markdown OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GIB = 2 ** 30


def _read_jsonl(path: Path) -> list:
    """Records of a JSON-lines file; lines that fail json.loads are skipped, a missing file means none."""
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _window(recs: list, lo: float, hi: float) -> list:
    """Records with lo < t <= hi."""
    return [r for r in recs if lo < r["t"] <= hi]


def _segment(grows: list, evicts: list) -> dict:
    """Grow and eviction counters for one segment."""
    dts = [g["dt"] for g in grows]
    return {"grows": sum(1 for g in grows if g.get("cap1") != g.get("cap0")),
            "refused": sum(1 for g in grows if g.get("ok") is not True),
            "slow_grows": sum(1 for g in grows if g["dt"] >= 1.0),
            "grow_s_max": max(dts) if dts else 0.0,
            "grow_s_sum": round(sum(dts), 3) if dts else 0.0,
            "evicts": sum(1 for e in evicts if e.get("ok") is True)}


def _diff(a, b, ndigits: int = 1):
    """round(a - b, ndigits), or None when either side is None."""
    return None if a is None or b is None else round(a - b, ndigits)


def digest(out_dir, tags, base_config: str = "off") -> dict:
    """THOR-DEC-1 memory digest: {"tags": {tag: entry}} for *tags* under *out_dir*."""
    out_dir = Path(out_dir)
    tags = list(tags)
    probes_in = {t: [] for t in tags}
    for row in _read_jsonl(out_dir / "probe.jsonl"):
        if row.get("tag") in probes_in:
            probes_in[row["tag"]].append((row["phase"], row["t"], float(row["memavail_mib"]),
                                          float(row["free_mib"])))
    entries = {}
    for tag in tags:
        recs = _read_jsonl(out_dir / f"diag_{tag}.jsonl")
        grows = [r for r in recs if r.get("k") == "grow"]
        evicts = [r for r in recs if r.get("k") == "evict"]
        mds = [r for r in recs if r.get("k") == "md"]
        room_gib = reserve_gib = window_gib = streams = None
        if mds:
            md = mds[-1]
            room_gib = round(md["room"] / GIB, 3)
            reserve_gib = round(md["reserve"] / GIB, 3)
            window_gib = round(md["window_bytes"] / GIB, 3)
            streams = int((md["room"] - md["reserve"]) // md["window_bytes"])
        plist = probes_in[tag]
        pre = next((p for p in plist if p[0] == "pre"), None)
        probes = {}
        for phase, _t, ma, fr in plist:
            probes[phase] = {"memavail_mib": ma, "free_mib": fr,
                             "foot_avail_mib": None if pre is None else round(pre[2] - ma, 1),
                             "foot_free_mib": None if pre is None else round(pre[3] - fr, 1)}
        segments = {}
        if plist or recs:
            for i in range(1, len(plist)):
                lo, hi = plist[i - 1][1], plist[i][1]
                segments[plist[i][0]] = _segment(_window(grows, lo, hi), _window(evicts, lo, hi))
            segments["all"] = _segment(grows, evicts)
        sts = [r for r in recs if r.get("k") == "st"]
        last_st = None
        if sts:
            last = sts[-1]
            last_st = {"waits": last.get("waits", 0), "ends": last.get("ends", 0),
                       "retries": last.get("retries", 0)}
        base = None
        if "_" in tag:
            cand = f"{base_config}_{tag.rsplit('_', 1)[1]}"
            if cand in tags and cand != tag:
                base = cand
        entries[tag] = {"room_gib": room_gib, "reserve_gib": reserve_gib, "window_gib": window_gib,
                        "full_window_streams": streams, "probes": probes, "segments": segments,
                        "last_st": last_st, "base": base, "vs_base": None}
    for entry in entries.values():
        if entry["base"] is None:
            continue
        b = entries[entry["base"]]
        vs = {"room_gib": _diff(entry["room_gib"], b["room_gib"], 3)}
        for phase, p in entry["probes"].items():
            if phase == "pre" or phase not in b["probes"]:
                continue
            bp = b["probes"][phase]
            vs[phase] = {"foot_avail_mib": _diff(p["foot_avail_mib"], bp["foot_avail_mib"]),
                         "foot_free_mib": _diff(p["foot_free_mib"], bp["foot_free_mib"])}
        entry["vs_base"] = vs
    return {"tags": entries}


def _cell(v) -> str:
    """Table cell text: '-' for None."""
    return "-" if v is None else str(v)


def _pm(v) -> str:
    """Signed one-decimal table cell: '-' for None."""
    return "-" if v is None else f"{v:+.1f}"


def markdown(d: dict) -> str:
    """Render the digest as Markdown; the result ends with a newline."""
    lines = ["# THOR-DEC-1 memory digest",
             "| tag | room GiB | full-window streams | phase | MemAvailable MiB | driver free MiB "
             "| footprint avail MiB | footprint free MiB | vs base avail | vs base free |"]
    for tag, entry in d["tags"].items():
        vs = entry["vs_base"] or {}
        for phase, p in entry["probes"].items():
            pv = vs.get(phase) or {}
            cells = [tag, _cell(entry["room_gib"]), _cell(entry["full_window_streams"]), phase,
                     _cell(p["memavail_mib"]), _cell(p["free_mib"]), _cell(p["foot_avail_mib"]),
                     _cell(p["foot_free_mib"]), _pm(pv.get("foot_avail_mib")),
                     _pm(pv.get("foot_free_mib"))]
            lines.append("| " + " | ".join(cells) + " |")
    items = [f"{tag} {e['vs_base']['room_gib']:+.3f}" for tag, e in d["tags"].items()
             if e["vs_base"] and e["vs_base"]["room_gib"] is not None]
    if items:
        lines.append("room vs base GiB: " + ", ".join(items))
    lines.append("| tag | segment | grows | refused | slow grows | grow s max | grow s sum | evicts |")
    for tag, entry in d["tags"].items():
        for seg, s in entry["segments"].items():
            lines.append("| " + " | ".join([tag, seg, str(s["grows"]), str(s["refused"]),
                                            str(s["slow_grows"]), _cell(s["grow_s_max"]),
                                            _cell(s["grow_s_sum"]), str(s["evicts"])]) + " |")
    lines.append("| tag | waits | ends | retries |")
    for tag, entry in d["tags"].items():
        st = entry["last_st"]
        if st is None:
            cells = [tag, "-", "-", "-"]
        else:
            cells = [tag, st["waits"], st["ends"], st["retries"]]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    """CLI entry point; returns 0."""
    ap = argparse.ArgumentParser(description="THOR-DEC-1 memory digest of a run's out folder.")
    ap.add_argument("out_dir", type=Path, help="run out folder with probe.jsonl and diag_<tag>.jsonl")
    ap.add_argument("--tags", required=True, help="comma-separated server tags")
    ap.add_argument("--base-config", default="off", help="base config name for the same-round diff")
    ap.add_argument("--json", type=Path, default=None, help="also write the digest as JSON here")
    ap.add_argument("--markdown", type=Path, default=None, help="also write the digest as Markdown here")
    args = ap.parse_args(argv)
    tags = [t for t in args.tags.split(",") if t]
    d = digest(args.out_dir, tags, base_config=args.base_config)
    text = markdown(d)
    print(text, end="")
    if args.json is not None:
        args.json.write_text(json.dumps(d, indent=1), encoding="utf-8")
    if args.markdown is not None:
        args.markdown.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
