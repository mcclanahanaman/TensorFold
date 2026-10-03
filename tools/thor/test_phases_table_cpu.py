"""Spec of phases_table.py's host-phase and CPU columns and ``--skip-first`` (THOR-DEC-1, decode_phases ``--cpu``)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import phases_table as pt  # noqa: E402

PH = ("stage", "tables", "verify", "sample", "commit", "mtp_stage", "mtp", "picks")
CPU_A = [["start", 2, 2601000, 50.0], ["sample", 2, 972000, 150.0], ["picks", 4, 1497000, 90.0]]
CPU_B = [["start", 2, 2401000, 70.0], ["sample", 3, 1172000, 110.0], ["picks", 4, 1703000, 70.0]]


def rnd(wall, host, cpu=None):
    """Phase i (in PH order) takes host * (i + 1) ms on the host and 0.5 * (i + 1) ms on the GPU."""

    phases = {p: [host * (i + 1), 0.5 * (i + 1), 1] for i, p in enumerate(PH)}
    r = {"wall_ms": wall, "span_ms": wall, "gap_ms": 1.0, "rows": 4, "phases": phases,
         "graph": {"captures": 0, "replays": 0, "capture_s": 0.0}}
    if cpu is not None:
        r["cpu"] = cpu
    return r


def group(users, kind, g, tok_s, rounds):
    return {"users": users, "type": kind, "group": g, "agg_tok_s": tok_s, "tokens_per_round": 3.0, "rounds": rounds,
            "decode_tokens": 100, "out": [101] * users}


def report(label, cpu=True):
    a, b = (CPU_A, CPU_B) if cpu else (None, None)
    return {"label": label, "groups": [
        group(1, "prose", 0, 40.0, [rnd(80.0, 1.0, a)]),                      # the warm-up group
        group(1, "prose", 1, 50.0, [rnd(60.0, 0.5, a), rnd(62.0, 0.5, b)]),
        group(1, "prose", 2, 54.0, [rnd(58.0, 0.5, b)]),
        group(4, "code", 0, 120.0, [rnd(100.0, 2.0, a)])]}                     # its only group: kept by skip_first


def test_host_phase_and_cpu_columns():
    rows = pt.table({"on": report("on")}, baseline="on")
    r = rows[0]
    assert (r["label"], r["users"], r["type"], r["groups"], r["rounds"]) == ("on", 1, "prose", 3, 4)
    assert r["stage_host_ms"] == pytest.approx(0.625)                # mean over the pooled rounds
    assert r["tables_host_ms"] == pytest.approx(1.25)
    assert r["commit_host_ms"] == pytest.approx(3.125)
    assert r["mtp_stage_host_ms"] == pytest.approx(3.75)
    assert r["picks_host_ms"] == pytest.approx(5.0)
    assert r["sample_gpu_ms"] == pytest.approx(2.0)
    assert r["cpu_mhz"] == pytest.approx({"start": 2501.0, "sample": 1072.0, "picks": 1600.0})
    assert r["probe_us"] == pytest.approx({"start": 60.0, "sample": 130.0, "picks": 80.0})


def test_skip_first_drops_each_cells_first_group():
    rows = pt.table({"on": report("on")}, baseline="on", skip_first=True)
    prose, code = rows
    assert prose["groups"] == 2 and prose["rounds"] == 3
    assert prose["agg_tok_s"] == pytest.approx(52.0)
    assert prose["wall_ms"] == pytest.approx(60.0)
    assert prose["stage_host_ms"] == pytest.approx(0.5)
    assert prose["cpu_mhz"]["start"] == pytest.approx((2601 + 2401 + 2401) / 3)
    assert prose["cpu_mhz"]["sample"] == pytest.approx((972 + 1172 + 1172) / 3)
    assert code["groups"] == 1 and code["rounds"] == 1                # a cell with one group keeps it
    assert code["agg_tok_s"] == pytest.approx(120.0)


def test_markdown_cpu_columns_only_with_cpu_data():
    with_cpu = pt.markdown(pt.table({"on": report("on")}, baseline="on"))
    head = with_cpu.splitlines()[0]
    assert "MHz start/sample/picks" in head and "probe us start/sample/picks" in head
    assert "stage host" in head and "commit host" in head
    assert "| 2501/1072/1600 |" in with_cpu and "| 60/130/80 |" in with_cpu
    without = pt.markdown(pt.table({"off": report("off", cpu=False)}, baseline="off"))
    assert "MHz" not in without.splitlines()[0]
    rows = pt.table({"off": report("off", cpu=False)}, baseline="off")
    assert rows[0].get("cpu_mhz") in (None, {})
    mixed = pt.markdown(pt.table({"off": report("off", cpu=False), "on": report("on")}, baseline="off"))
    off_line = [line for line in mixed.splitlines() if line.startswith("| off | 1 | prose |")][0]
    assert "| -/-/- |" in off_line                                    # a row without samples, in a table with them


def test_cli_skip_first(tmp_path):
    a = tmp_path / "ph_on.json"
    a.write_text(json.dumps(report("on")))
    out = tmp_path / "t.json"
    subprocess.run([sys.executable, str(Path(pt.__file__)), str(a), "--baseline", "on", "--skip-first", "--json",
                    str(out)], capture_output=True, text=True, check=True)
    rows = json.loads(out.read_text())
    assert [r["groups"] for r in rows] == [2, 1]
