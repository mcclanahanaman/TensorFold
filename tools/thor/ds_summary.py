"""DEC-STRUCT tables from ds_drive JSON runs: a run is a dict {label, records} whose records carry the phase
('ref1' | 'wave' | 'ref2'), the request coordinates (type, conc, rep, stream, prompt), the reply (text,
reasoning, error, same_as_ref1, diff_at; full text = reasoning + text) and the per-request MTP stats in d
(comp, rounds, drafted, accepted, cached, tpr, acc, dpr, any value possibly None); ds_summary turns such runs
into the DEC-STRUCT comparison tables — wave streams against their 1-stream reference, per-(type, conc) means,
ref2-vs-ref1 stability and cross-run ref1 identity — as row dicts, as markdown and via a small CLI."""

import argparse
import json
import statistics
from pathlib import Path

_WAVE_KEYS = ("type", "conc", "rep", "stream", "tok", "rounds", "tpr", "acc", "dpr", "cached", "same",
              "ref_tpr", "ref_acc")
_CELL_KEYS = ("type", "conc", "streams", "identical", "tpr_mean", "tpr_min", "tpr_max", "ref_tpr_mean")


def _full_text(rec):
    return rec["reasoning"] + rec["text"]


def _ref1_by_prompt(run):
    refs = {}
    for rec in run["records"]:
        if rec["phase"] == "ref1" and rec["prompt"] not in refs:
            refs[rec["prompt"]] = rec
    return refs


def wave_rows(run):
    refs = _ref1_by_prompt(run)
    rows = []
    for rec in run["records"]:
        if rec["phase"] != "wave":
            continue
        d = rec["d"]
        ref = refs.get(rec["prompt"])
        rows.append({"label": run["label"], "type": rec["type"], "conc": rec["conc"], "rep": rec["rep"],
                     "stream": rec["stream"], "tok": d["comp"], "rounds": d["rounds"], "tpr": d["tpr"],
                     "acc": d["acc"], "dpr": d["dpr"], "cached": d["cached"], "same": rec["same_as_ref1"],
                     "diff_at": rec["diff_at"], "ref_tpr": ref["d"]["tpr"] if ref else None,
                     "ref_acc": ref["d"]["acc"] if ref else None})
    return rows


def cell_summary(run):
    refs = _ref1_by_prompt(run)
    cells, order = {}, []
    for rec in run["records"]:
        if rec["phase"] != "wave":
            continue
        key = (rec["type"], rec["conc"])
        if key not in cells:
            cells[key] = {"streams": 0, "identical": 0, "tprs": [], "ref_tprs": []}
            order.append(key)
        cell = cells[key]
        cell["streams"] += 1
        if rec["same_as_ref1"] is True:
            cell["identical"] += 1
        if rec["d"]["tpr"] is not None:
            cell["tprs"].append(rec["d"]["tpr"])
        ref = refs.get(rec["prompt"])
        if ref is not None and ref["d"]["tpr"] is not None:
            cell["ref_tprs"].append(ref["d"]["tpr"])
    rows = []
    for key in order:
        cell = cells[key]
        rows.append({"type": key[0], "conc": key[1], "streams": cell["streams"], "identical": cell["identical"],
                     "tpr_mean": round(statistics.fmean(cell["tprs"]), 3) if cell["tprs"] else None,
                     "tpr_min": min(cell["tprs"]) if cell["tprs"] else None,
                     "tpr_max": max(cell["tprs"]) if cell["tprs"] else None,
                     "ref_tpr_mean": round(statistics.fmean(cell["ref_tprs"]), 3) if cell["ref_tprs"] else None})
    return rows


def ref_stability(run):
    refs = _ref1_by_prompt(run)
    rows = []
    for rec in run["records"]:
        if rec["phase"] != "ref2":
            continue
        ref = refs.get(rec["prompt"])
        rows.append({"label": run["label"], "type": rec["type"], "prompt": rec["prompt"],
                     "same": rec["same_as_ref1"], "cached": rec["d"]["cached"],
                     "tpr1": ref["d"]["tpr"] if ref else None, "tpr2": rec["d"]["tpr"]})
    return rows


def _first_diff(texts):
    base = texts[0]
    for other in texts[1:]:
        if other == base:
            continue
        for i in range(min(len(base), len(other))):
            if base[i] != other[i]:
                return i
        return min(len(base), len(other))
    return -1


def cross(runs):
    ref_maps = [_ref1_by_prompt(run) for run in runs]
    rows, seen = [], set()
    for rec in ref_maps[0].values():
        prompt = rec["prompt"]
        if prompt in seen or any(prompt not in m for m in ref_maps):
            continue
        seen.add(prompt)
        texts = [_full_text(m[prompt]) for m in ref_maps]
        rows.append({"type": rec["type"], "prompt": prompt,
                     "labels": [run["label"] for run in runs],
                     "tpr": [m[prompt]["d"]["tpr"] for m in ref_maps],
                     "identical": all(t == texts[0] for t in texts[1:]),
                     "diff_at": _first_diff(texts)})
    return rows


def snippet(text, n=60):
    text = text.replace("\n", "\\n")
    if len(text) <= 2 * n:
        return text
    return text[:n] + " ... " + text[-n:]


def _fmt(value):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _tail(text):
    return (text or "")[-16:]


def _line(cells):
    return "| " + " | ".join(cells) + " |"


def _table(headers, rows):
    return [_line(headers), _line(["---"] * len(headers))] + [_line(row) for row in rows]


def markdown(runs):
    lines = []

    def section(title):
        if lines:
            lines.append("")
        lines.append(title)

    for run in runs:
        section(f"## {run['label']}: wave streams")
        lines += _table(("type", "conc", "rep", "stream", "tok", "rounds", "tpr", "acc", "dpr", "cached",
                         "same", "ref tpr", "ref acc"),
                        [[_fmt(row[k]) for k in _WAVE_KEYS] for row in wave_rows(run)])
        section(f"## {run['label']}: per cell")
        lines += _table(("type", "conc", "streams", "identical", "tpr mean", "tpr min", "tpr max",
                         "ref tpr mean"),
                        [[_fmt(row[k]) for k in _CELL_KEYS] for row in cell_summary(run)])
    section("## references: ref2 vs ref1")
    lines += _table(("run", "type", "prompt", "same", "cached", "tpr ref1", "tpr ref2"),
                    [[row["label"], _fmt(row["type"]), _tail(row["prompt"]), _fmt(row["same"]),
                      _fmt(row["cached"]), _fmt(row["tpr1"]), _fmt(row["tpr2"])]
                     for run in runs for row in ref_stability(run)])
    section("## ref1 across runs")
    lines += _table(("type", "prompt", "identical", "diff_at", "tpr per run"),
                    [[_fmt(row["type"]), _tail(row["prompt"]), _fmt(row["identical"]), _fmt(row["diff_at"]),
                      " / ".join(_fmt(t) for t in row["tpr"])] for row in cross(runs)])
    section("## structured ref1 text")
    for run in runs:
        for rec in run["records"]:
            if rec["phase"] == "ref1" and rec["type"] == "structured":
                lines.append(f"- {run['label']} {_tail(rec['prompt'])}: `{snippet(_full_text(rec), 60)}`")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="DEC-STRUCT tables from ds_drive JSON runs.")
    ap.add_argument("files", nargs="+", metavar="JSON", help="ds_drive run JSON files")
    ap.add_argument("--md", metavar="PATH", help="write the markdown report here")
    ap.add_argument("--json", metavar="PATH", help="write the combined JSON summary here")
    args = ap.parse_args()
    runs = [json.loads(Path(path).read_text()) for path in args.files]
    md = markdown(runs)
    print(md, end="")
    if args.md:
        Path(args.md).write_text(md)
    if args.json:
        payload = {"waves": [row for run in runs for row in wave_rows(run)],
                   "cells": [dict(row, label=run["label"]) for run in runs for row in cell_summary(run)],
                   "refs": [row for run in runs for row in ref_stability(run)],
                   "cross": cross(runs)}
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
