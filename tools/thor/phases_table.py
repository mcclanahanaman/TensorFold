"""tools/thor/phases_table.py: decode_phases.py reports (one per config) -> one row per (label, users, type) pooling
its groups, with per-phase host/GPU means over the pooled rounds, graph-counter sums and a tok/s ratio against a
baseline config, printed as markdown (THOR-DEC-1).

    python tools/thor/phases_table.py A.json [B.json ...] --baseline LABEL [--json OUT] [--markdown OUT]

agg_tok_s and tokens_per_round are means over a config's groups; wall_ms, gap_ms, mtp_steps and every phase's
host/GPU time are means over the pooled rounds (a round lacking a phase contributes zeros); captures, replays and
capture_s are sums. vs_baseline divides a row's agg_tok_s by the --baseline label's agg_tok_s at the same
(users, type) and is None when that cell is missing."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# decode_phases.PHASES copied (decode_phases imports torch; this module must stay stdlib-only).
PHASES = ("stage", "tables", "verify", "sample", "commit", "mtp_stage", "mtp", "picks")

_GRAPH_KEYS = ("captures", "replays", "capture_s")


def load(paths) -> dict:
    """Read report paths into {label: report}, in the given order."""
    reports: dict = {}
    for path in paths:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        reports[report["label"]] = report
    return reports


def _pool(groups: list) -> dict:
    rounds = [r for g in groups for r in g.get("rounds", [])]

    def over_rounds(i: int, phase: str) -> float:
        return statistics.fmean([r.get("phases", {}).get(phase, [0, 0, 0])[i] for r in rounds]) if rounds else 0.0

    row = {"groups": len(groups), "rounds": len(rounds),
           "agg_tok_s": statistics.fmean([g["agg_tok_s"] for g in groups]) if groups else 0.0,
           "tokens_per_round": statistics.fmean([g["tokens_per_round"] for g in groups]) if groups else 0.0,
           "wall_ms": statistics.fmean([r["wall_ms"] for r in rounds]) if rounds else 0.0,
           "gap_ms": statistics.fmean([r["gap_ms"] for r in rounds]) if rounds else 0.0}
    for phase in PHASES:
        row[f"{phase}_host_ms"] = over_rounds(0, phase)
        row[f"{phase}_gpu_ms"] = over_rounds(1, phase)
    row["mtp_steps"] = over_rounds(2, "mtp")
    for key in _GRAPH_KEYS:
        row[key] = sum(r.get("graph", {}).get(key, 0) for r in rounds)
    return row


def table(reports: dict, baseline: str) -> list:
    """One row per (label, users, type): configs in dict order, within a config sorted by (users, type)."""
    pooled: dict = {}                                 # (label, users, type) -> pooled row
    for label, report in reports.items():
        keyed: dict = {}
        for g in report.get("groups", []):
            keyed.setdefault((g["users"], g["type"]), []).append(g)
        for (users, kind), groups in keyed.items():
            row = {"label": label, "users": users, "type": kind, "baseline": baseline}
            row.update(_pool(groups))
            pooled[(label, users, kind)] = row
    rows = []
    for label, report in reports.items():
        for key in sorted(k for k in pooled if k[0] == label):
            row = pooled[key]
            base = pooled.get((baseline, key[1], key[2]))   # the baseline's pooled agg for the same (users, type)
            row["vs_baseline"] = row["agg_tok_s"] / base["agg_tok_s"] if base and base["agg_tok_s"] else None
            rows.append(row)
    return rows


def markdown(rows: list) -> str:
    """Markdown table: a header, a separator and one line per row."""
    if not rows:
        return ""
    head = ["config", "users", "type", "tok/s", f"vs {rows[0]['baseline']}", "tok/round", "round ms", "verify gpu",
            "verify host", "mtp gpu", "mtp steps", "picks gpu", "sample gpu", "gap", "captures", "replays"]
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]

    def cell(value, digits=None) -> str:
        return f"{value:.{digits}f}" if digits is not None else str(value)

    for r in rows:
        cells = [r["label"], r["users"], r["type"], cell(r["agg_tok_s"], 2),
                 cell(r["vs_baseline"], 3) if r["vs_baseline"] is not None else "-",
                 cell(r["tokens_per_round"], 2), cell(r["wall_ms"], 2), cell(r["verify_gpu_ms"], 2),
                 cell(r["verify_host_ms"], 2), cell(r["mtp_gpu_ms"], 2), cell(r["mtp_steps"], 2),
                 cell(r["picks_gpu_ms"], 2), cell(r["sample_gpu_ms"], 2), cell(r["gap_ms"], 2),
                 r["captures"], r["replays"]]
        lines.append("| " + " | ".join(cell(c) for c in cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+", type=Path)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--json", dest="json_out", type=Path)
    ap.add_argument("--markdown", dest="markdown_out", type=Path)
    a = ap.parse_args(argv)
    rows = table(load(a.reports), baseline=a.baseline)
    md = markdown(rows)
    if a.json_out is not None:
        a.json_out.write_text(json.dumps(rows, indent=2))
    if a.markdown_out is not None:
        a.markdown_out.write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
