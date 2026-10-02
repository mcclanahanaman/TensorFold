"""Tests for graph_stats.py: the digest of the ``[thor-graph]`` lines a Thor decode-graph server logs (THOR-K3).

Each such line is the prefix ``[thor-graph] `` and one JSON object. Events: ``on`` (graphs enabled: cap, min_seen,
free_mib), ``capture`` (kind, n, rows, graphs, capture_s, reserved_mib, free_mib), ``stats`` (cumulative: forwards,
replays, captures by kind, evictions, capture_s, graphs, reserved_mib, free_mib) and ``fail`` (kind, error).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import graph_stats as gs  # noqa: E402


def _line(obj) -> str:
    return "[thor-graph] " + json.dumps(obj)


ON = {"event": "on", "cap": 128, "min_seen": 2, "free_mib": 30000.0}
CAP1 = {"event": "capture", "kind": "verify", "n": 1, "rows": 3, "graphs": 1, "capture_s": 0.25,
        "reserved_mib": 1000.0, "free_mib": 29800.0}
CAP2 = {"event": "capture", "kind": "mtp", "n": 1, "rows": 1, "graphs": 2, "capture_s": 0.05,
        "reserved_mib": 1010.0, "free_mib": 29790.0}
CAP3 = {"event": "capture", "kind": "verify", "n": 2, "rows": 5, "graphs": 3, "capture_s": 0.5,
        "reserved_mib": 1020.0, "free_mib": 29700.0}
ST1 = {"event": "stats", "forwards": {"verify": 10, "mtp": 20}, "replays": {"verify": 5, "mtp": 15},
       "captures": {"verify": 1, "mtp": 1}, "evictions": 0, "capture_s": 0.3, "graphs": 2, "reserved_mib": 1010.0,
       "free_mib": 29790.0}
ST2 = {"event": "stats", "forwards": {"verify": 100, "mtp": 300}, "replays": {"verify": 90, "mtp": 297},
       "captures": {"verify": 2, "mtp": 1}, "evictions": 1, "capture_s": 0.8, "graphs": 3, "reserved_mib": 1020.0,
       "free_mib": 29700.0}
FAIL = {"event": "fail", "kind": "mtp", "error": "operation not permitted when stream is capturing"}

LOG = "\n".join([
    "INFO server starting",
    _line(ON),
    "[thor-graph] {not json",
    _line(CAP1),
    "some other line [thor-graph] in the middle is not an event",
    _line(CAP2),
    _line(ST1),
    _line(CAP3),
    _line(FAIL),
    _line(ST2),
    "INFO done",
])


def test_parse_keeps_events_in_order_and_counts_bad_lines():
    events, bad = gs.parse(LOG)
    assert [e["event"] for e in events] == ["on", "capture", "capture", "stats", "capture", "fail", "stats"]
    assert bad == 1                         # the prefixed line whose JSON does not parse
    assert events[1] == CAP1


def test_summary_takes_the_last_stats_and_derives_rates():
    s = gs.summarize(gs.parse(LOG)[0], label="serve_on_1")
    assert s["label"] == "serve_on_1"
    assert s["enabled"] is True
    assert s["forwards"] == {"verify": 100, "mtp": 300}
    assert s["replays"] == {"verify": 90, "mtp": 297}
    assert s["captures"] == {"verify": 2, "mtp": 1}
    assert s["hit_rate"] == {"verify": pytest.approx(0.9), "mtp": pytest.approx(0.99)}
    assert s["evictions"] == 1
    assert s["capture_s"] == pytest.approx(0.8)
    assert s["graphs"] == 3
    assert s["reserved_mib"] == pytest.approx(1020.0)
    assert s["free_drop_mib"] == pytest.approx(300.0)       # the 'on' event's free_mib less the last stats' one
    assert s["failures"] == 1
    assert s["fail_kinds"] == ["mtp"]
    # capture events by (kind, n): counts and the slowest capture
    assert s["captures_by"] == {"verify/1": 1, "mtp/1": 1, "verify/2": 1}
    assert s["max_capture_s"] == pytest.approx(0.5)


def test_summary_of_a_log_without_events_is_disabled():
    s = gs.summarize(gs.parse("INFO nothing here\n")[0], label="serve_off_1")
    assert s["enabled"] is False
    assert s["forwards"] == {} and s["replays"] == {} and s["captures"] == {}
    assert s["hit_rate"] == {}
    assert s["graphs"] == 0 and s["failures"] == 0
    assert s["free_drop_mib"] is None and s["reserved_mib"] is None


def test_summary_without_stats_uses_capture_events_only():
    text = "\n".join([_line(ON), _line(CAP1)])
    s = gs.summarize(gs.parse(text)[0], label="x")
    assert s["enabled"] is True
    assert s["forwards"] == {} and s["hit_rate"] == {}
    assert s["graphs"] == 1                                   # the last capture's graph count
    assert s["captures_by"] == {"verify/1": 1}
    assert s["free_drop_mib"] == pytest.approx(200.0)         # 'on' less the last capture's free_mib
    assert s["reserved_mib"] == pytest.approx(1000.0)


def test_hit_rate_of_a_kind_with_no_forwards_is_none():
    st = dict(ST2, forwards={"verify": 0, "mtp": 4}, replays={"verify": 0, "mtp": 2})
    s = gs.summarize(gs.parse("\n".join([_line(ON), _line(st)]))[0], label="y")
    assert s["hit_rate"] == {"verify": None, "mtp": pytest.approx(0.5)}


def test_markdown_has_one_row_a_log(tmp_path):
    rows = [gs.summarize(gs.parse(LOG)[0], label="serve_on_1"),
            gs.summarize(gs.parse("plain\n")[0], label="serve_off_1")]
    md = gs.markdown(rows)
    lines = md.strip().splitlines()
    assert lines[0] == ("| log | graphs on | verify hit | mtp hit | captures | capture s | graphs | evictions "
                        "| reserved MiB | free drop MiB | failures |")
    assert lines[1] == "|---|---|---|---|---|---|---|---|---|---|---|"
    assert len(lines) == 4
    assert lines[2].startswith("| serve_on_1 | yes | 90.0% | 99.0% | 3 | 0.80 | 3 | 1 | 1,020.0 | 300.0 | 1 |")
    assert lines[3] == "| serve_off_1 | no | - | - | 0 | - | 0 | - | - | - | 0 |"


def test_main_reads_logs_and_writes_json(tmp_path, capsys):
    a = tmp_path / "serve_on_1.log"
    a.write_text(LOG)
    b = tmp_path / "serve_off_1.log"
    b.write_text("INFO\n")
    out = tmp_path / "stats.json"
    assert gs.main([str(a), str(b), "--json", str(out)]) == 0
    data = json.loads(out.read_text())
    assert [r["label"] for r in data["logs"]] == ["serve_on_1", "serve_off_1"]
    assert data["logs"][0]["bad_lines"] == 1
    printed = capsys.readouterr().out
    assert "| serve_on_1 | yes |" in printed
