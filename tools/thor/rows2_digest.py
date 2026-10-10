"""ROWS-2 per-request digest: host memory, n-gram table residency, faults and server counters
before and after each request.

Usage:
    python tools/thor/rows2_digest.py OUT_DIR --tags T1,T2 [--json OUT] [--markdown OUT]

Per tag TAG this reads three inputs from OUT_DIR:
- marks_<tag>.jsonl (from marks_run.py) lists the request windows, one mark per request with tool, seq and t0/t1.
- mw_<tag>.jsonl (from memwatch.py) holds periodic host samples: mem, vm, psi, proc and optional table residency.
- diag_<tag>.jsonl (from serve_diag.py) holds server records: start/md headers, st counter snapshots, pass records.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NATURAL = ["nat_cold", "nat_warm"]
SERVE_AB = ["p8k_0", "p8k_1", "p32k_0", "p32k_1", "p128k_0", "p128k_1", "f60k_1", "f60k_2",
            "d1", "d4", "d4", "d4", "d4"]
GIB = 2 ** 30
KB_GIB = 1048576
ROW_KEYS = ["label", "n", "t0", "dur_s", "free_before", "free_after", "avail_before", "avail_after",
            "cached_before", "cached_after", "table_before", "table_after", "majflt", "refault", "pgscan",
            "pgsteal", "allocstall", "psi_some_ms", "psi_full_ms", "g_s", "g_rows", "stage_s", "comp_s",
            "pass_s", "passes", "rounds", "round_s", "grows", "evicts", "ec_calls", "ec_s", "dev_allocs",
            "held_gib", "kept", "kept_rows", "reserved_gib", "pass_n", "pass_max_s", "pass_majflt", "gpu_s"]
DETAIL = ["dur_s", "table_before", "table_after", "majflt", "refault", "pgsteal", "g_s", "stage_s", "pass_s",
          "gpu_s", "free_before", "avail_before", "held_gib", "kept", "evicts"]
COMPARE = ["dur_s", "table_before", "table_after", "majflt", "refault", "pgsteal", "g_s", "stage_s", "pass_s",
           "gpu_s", "free_before", "avail_before", "held_gib", "kept"]
SCHED = {"natural_prefill": NATURAL, "serve_ab": SERVE_AB}

_ST_FLOAT = ("g_s", "stage_s", "comp_s", "pass_s", "round_s", "ec_s")
_ST_INT = ("g_rows", "passes", "rounds", "grows", "evicts", "ec_calls", "dev_allocs")
_ST_AFTER = _ST_FLOAT + _ST_INT + ("held_gib", "kept", "kept_rows", "reserved_gib")
_HOST_KEYS = ("free_before", "free_after", "avail_before", "avail_after", "cached_before", "cached_after",
              "table_before", "table_after", "majflt", "refault", "pgscan", "pgsteal", "allocstall",
              "psi_some_ms", "psi_full_ms")


def load_jsonl(path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict):
                records.append(rec)
    return records


def _label(tool, seq):
    lst = SCHED.get(tool)
    if lst is not None and isinstance(seq, int) and 0 <= seq < len(lst):
        return lst[seq]
    return f"{tool}_{seq}"


def windows(marks) -> list[dict]:
    out = []
    for m in marks:
        label = _label(m.get("tool"), m.get("seq"))
        if out and out[-1]["label"] == label:
            out[-1]["t0"] = min(out[-1]["t0"], m["t0"])
            out[-1]["t1"] = max(out[-1]["t1"], m["t1"])
            out[-1]["n"] += 1
        else:
            out.append({"label": label, "n": 1, "t0": m["t0"], "t1": m["t1"]})
    return out


def _before(samples, t0):
    for s in reversed(samples):
        if s["t"] <= t0:
            return s
    return samples[0]


def _after(samples, t1):
    for s in samples:
        if s["t"] >= t1:
            return s
    return samples[-1]


def _vm_val(rec, key):
    vm = rec.get("vm") or {}
    return vm.get(key, 0)


def _allocstall(rec):
    vm = rec.get("vm") or {}
    return sum(v for k, v in vm.items() if str(k).startswith("allocstall"))


def _psi_delta(before, after, key):
    pb, pa = before.get("psi"), after.get("psi")
    if not isinstance(pb, dict) or key not in pb or not isinstance(pa, dict) or key not in pa:
        return None
    return round((pa[key] - pb[key]) / 1000, 1)


def _host_row(samples, w):
    if not samples:
        return {k: None for k in _HOST_KEYS}
    b, a = _before(samples, w["t0"]), _after(samples, w["t1"])
    row = {}
    for name, key in (("free", "MemFree"), ("avail", "MemAvailable"), ("cached", "Cached")):
        row[name + "_before"] = round(b["mem"].get(key, 0) / KB_GIB, 2)
        row[name + "_after"] = round(a["mem"].get(key, 0) / KB_GIB, 2)
    tsamples = [s for s in samples if s.get("table") is not None]
    row["table_before"] = round(100 * _before(tsamples, w["t0"])["table"], 1) if tsamples else None
    row["table_after"] = round(100 * _after(tsamples, w["t1"])["table"], 1) if tsamples else None
    row["majflt"] = a.get("proc", {}).get("majflt", 0) - b.get("proc", {}).get("majflt", 0)
    row["refault"] = _vm_val(a, "workingset_refault_file") - _vm_val(b, "workingset_refault_file")
    row["pgscan"] = (_vm_val(a, "pgscan_kswapd") + _vm_val(a, "pgscan_direct")
                     - _vm_val(b, "pgscan_kswapd") - _vm_val(b, "pgscan_direct"))
    row["pgsteal"] = (_vm_val(a, "pgsteal_kswapd") + _vm_val(a, "pgsteal_direct")
                      - _vm_val(b, "pgsteal_kswapd") - _vm_val(b, "pgsteal_direct"))
    row["allocstall"] = _allocstall(a) - _allocstall(b)
    row["psi_some_ms"] = _psi_delta(b, a, "some")
    row["psi_full_ms"] = _psi_delta(b, a, "full")
    return row


def _passes_in(diag, w):
    t0, t1 = w["t0"], w["t1"]
    return [r for r in diag if r.get("k") == "pass" and r.get("t0") is not None
            and t0 <= r["t0"] <= t1]


def _server_row(diag, w):
    row = {k: None for k in _ST_AFTER}
    row["pass_n"], row["pass_max_s"], row["pass_majflt"], row["gpu_s"] = 0, None, 0, None
    passes = _passes_in(diag, w)
    if passes:
        row["pass_n"] = len(passes)
        row["pass_max_s"] = round(max(r.get("dt", 0.0) for r in passes), 3)
        row["pass_majflt"] = sum(r.get("majflt", 0) for r in passes)
    ids = {r.get("i") for r in passes}
    ms = [r.get("ms", 0.0) for r in diag if r.get("k") == "pass_gpu" and r.get("i") in ids]
    if ms:
        row["gpu_s"] = round(sum(ms) / 1000, 3)
    sts = [r for r in diag if r.get("k") == "st"]
    if not sts:
        return row
    b, a = _before(sts, w["t0"]), _after(sts, w["t1"])
    for k in _ST_FLOAT:
        row[k] = round(a.get(k, 0) - b.get(k, 0), 3)
    for k in _ST_INT:
        row[k] = int(a.get(k, 0) - b.get(k, 0))
    row["held_gib"] = round(a.get("held", 0) / GIB, 2)
    row["kept"] = a.get("kept")
    row["kept_rows"] = a.get("kept_rows")
    row["reserved_gib"] = round(a.get("reserved", 0) / GIB, 2)
    return row


def digest_tag(folder, tag) -> dict:
    folder = Path(folder)
    marks = load_jsonl(folder / f"marks_{tag}.jsonl")
    mw = load_jsonl(folder / f"mw_{tag}.jsonl")
    diag = load_jsonl(folder / f"diag_{tag}.jsonl")
    samples = [r for r in mw if "mem" in r]
    starts = [r for r in diag if r.get("k") == "start"]
    mds = [r for r in diag if r.get("k") == "md"]
    md = mds[0] if mds else None
    requests = []
    for w in windows(marks):
        full = {"label": w["label"], "n": w["n"], "t0": w["t0"], "dur_s": round(w["t1"] - w["t0"], 3)}
        full.update(_host_row(samples, w))
        full.update(_server_row(diag, w))
        requests.append({k: full[k] for k in ROW_KEYS})
    return {"tag": tag,
            "ballast_gib": starts[0].get("ballast_gib", 0.0) if starts else 0.0,
            "prefill_rows": md.get("prefill_rows") if md else None,
            "room_gib": round(md["room"] / GIB, 2) if md and md.get("room") is not None else None,
            "reserve_gib": round(md["reserve"] / GIB, 2) if md and md.get("reserve") is not None else None,
            "requests": requests}


def _row_for(digest, label):
    for r in digest["requests"]:
        if r["label"] == label:
            return r
    return None


def compare(digests, metrics) -> list[dict]:
    labels = []
    for d in digests:
        for r in d["requests"]:
            if r["label"] not in labels:
                labels.append(r["label"])
    out = []
    for label in labels:
        entry = {"label": label}
        for metric in metrics:
            entry[metric] = {d["tag"]: (_row_for(d, label) or {}).get(metric) for d in digests}
        out.append(entry)
    return out


def _fmt(v):
    return "-" if v is None else str(v)


def markdown(digests, metrics) -> str:
    tags = [d["tag"] for d in digests]
    entries = compare(digests, metrics)
    lines = []
    for metric in metrics:
        lines.append(f"## {metric}")
        lines.append("")
        lines.append("| label | " + " | ".join(tags) + " |")
        lines.append("|" + "---|" * (1 + len(tags)))
        for e in entries:
            lines.append("| " + " | ".join([e["label"]] + [_fmt(e[metric][t]) for t in tags]) + " |")
        lines.append("")
    for d in digests:
        lines.append(f"## {d['tag']} (ballast {d['ballast_gib']} GiB, prefill rows {d['prefill_rows']}, "
                     f"room {d['room_gib']} GiB, reserve {d['reserve_gib']} GiB)")
        lines.append("")
        lines.append("| label | " + " | ".join(DETAIL) + " |")
        lines.append("|" + "---|" * (1 + len(DETAIL)))
        for r in d["requests"]:
            lines.append("| " + " | ".join([r["label"]] + [_fmt(r[k]) for k in DETAIL]) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ROWS-2 per-request digest")
    ap.add_argument("out_dir", type=Path, help="folder holding marks_/mw_/diag_<tag>.jsonl")
    ap.add_argument("--tags", required=True, help="comma-separated tags to digest")
    ap.add_argument("--json", type=Path, default=None, help="write the JSON digest here")
    ap.add_argument("--markdown", type=Path, default=None, help="write the Markdown digest here")
    ap.add_argument("--metrics", default=",".join(COMPARE), help="comma-separated metrics to compare")
    args = ap.parse_args(argv)
    tags = args.tags.split(",")
    metrics = args.metrics.split(",")
    digests = []
    for tag in tags:
        try:
            digest = digest_tag(args.out_dir, tag)
        except FileNotFoundError as exc:
            print(f"rows2_digest: missing input for tag {tag} in {args.out_dir}: {exc}", file=sys.stderr)
            return 1
        digests.append(digest)
        reqs = digest["requests"]
        print(f"{tag}: {len(reqs)} requests, " + ", ".join(f"{r['label']} {r['dur_s']} s" for r in reqs))
    if args.json is not None:
        payload = json.dumps({"digests": digests, "compare": compare(digests, metrics)}, indent=1)
        args.json.write_text(payload, encoding="utf-8")
    if args.markdown is not None:
        args.markdown.write_text(markdown(digests, metrics), encoding="utf-8")
    return 0


if __name__ == '__main__':
    sys.exit(main())
