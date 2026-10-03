"""Spec of tools/thor/phases_table.py: decode_phases.py reports (one per config) as a table (THOR-DEC-1)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import phases_table as pt  # noqa: E402

PH = ("stage", "tables", "verify", "sample", "commit", "mtp_stage", "mtp", "picks")


def rnd(wall, verify_gpu, mtp_gpu, steps, gap, captures=0, replays=0, capture_s=0.0):
    phases = {p: [0.1, 0.2, 1] for p in PH}
    phases["verify"] = [1.0, verify_gpu, 1]
    phases["mtp"] = [0.5 * steps, mtp_gpu, steps]
    return {"wall_ms": wall, "span_ms": wall, "gap_ms": gap, "rows": 4, "phases": phases,
            "graph": {"captures": captures, "replays": replays, "capture_s": capture_s}}


def group(users, kind, g, tok_s, tpr, rounds):
    return {"users": users, "type": kind, "group": g, "agg_tok_s": tok_s, "tokens_per_round": tpr, "rounds": rounds,
            "decode_tokens": 100, "out": [101] * users}


def report(label, scale):
    return {"label": label, "groups": [
        group(1, "prose", 0, 50.0 * scale, 3.0, [rnd(60 / scale, 40, 10, 4, 2.0), rnd(62 / scale, 41, 11, 4, 3.0)]),
        group(1, "prose", 1, 52.0 * scale, 3.4, [rnd(58 / scale, 39, 9, 2, 1.0, 2, 1, 0.5)]),
        group(4, "code", 0, 120.0 * scale, 5.0, [rnd(100, 70, 20, 5, 4.0, 0, 6)])]}


def test_rows_pool_groups_and_rounds(tmp_path):
    rows = pt.table({"off": report("off", 1.0), "on": report("on", 1.25)}, baseline="off")
    assert [(r["label"], r["users"], r["type"]) for r in rows] == [
        ("off", 1, "prose"), ("off", 4, "code"), ("on", 1, "prose"), ("on", 4, "code")]
    off1 = rows[0]
    assert off1["groups"] == 2 and off1["rounds"] == 3
    assert off1["agg_tok_s"] == pytest.approx(51.0)               # mean over groups
    assert off1["tokens_per_round"] == pytest.approx(3.2)
    assert off1["wall_ms"] == pytest.approx(60.0)                 # mean over the pooled rounds
    assert off1["verify_gpu_ms"] == pytest.approx(40.0)
    assert off1["verify_host_ms"] == pytest.approx(1.0)
    assert off1["mtp_gpu_ms"] == pytest.approx(10.0)
    assert off1["mtp_steps"] == pytest.approx(10 / 3)
    assert off1["gap_ms"] == pytest.approx(2.0)
    assert off1["picks_gpu_ms"] == pytest.approx(0.2)
    assert off1["captures"] == 2 and off1["replays"] == 1 and off1["capture_s"] == pytest.approx(0.5)
    assert off1["vs_baseline"] == pytest.approx(1.0)
    assert rows[2]["vs_baseline"] == pytest.approx(1.25)
    assert rows[3]["vs_baseline"] == pytest.approx(1.25)


def test_markdown_and_cli(tmp_path):
    a, b = tmp_path / "ph_off.json", tmp_path / "ph_on.json"
    a.write_text(json.dumps(report("off", 1.0)))
    b.write_text(json.dumps(report("on", 0.8)))
    md = pt.markdown(pt.table(pt.load([a, b]), baseline="off"))
    head = md.splitlines()[0]
    for col in ("config", "users", "type", "tok/s", "vs off", "tok/round", "round ms", "verify gpu", "mtp gpu",
                "gap", "captures"):
        assert col in head
    assert len(md.splitlines()) == 2 + 4
    assert "0.800" in md                                          # vs baseline, 3 decimals
    out = tmp_path / "t.json"
    got = subprocess.run([sys.executable, str(Path(pt.__file__)), str(a), str(b), "--baseline", "off", "--json",
                          str(out)], capture_output=True, text=True, check=True)
    assert "| on | 1 | prose |" in got.stdout
    assert len(json.loads(out.read_text())) == 4
