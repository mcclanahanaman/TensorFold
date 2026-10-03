"""Spec for tools/thor/stress_digest.py: the THOR-DEC-1 room and after-traffic memory digest. CPU only."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROBES = [("off_1", "pre", 100, 60000, 59000), ("off_1", "idle", 200, 36000, 35000),
          ("off_1", "drive", 300, 35000, 34100), ("off_1", "stress", 400, 34000, 33000),
          ("g_1", "pre", 500, 60100, 59050), ("g_1", "idle", 600, 35500, 34500),
          ("g_1", "drive", 700, 35050, 34100), ("g_1", "stress", 800, 34000, 33000)]
OFF = [{"k": "start", "t": 150.0},
       {"k": "md", "t": 190.0, "room": 26_200_000_000, "reserve": 2_147_483_648, "window_bytes": 4_800_000_000},
       {"k": "grow", "t": 250.0, "cap0": 8192, "cap1": 16384, "ok": True, "dt": 0.2},
       {"k": "grow", "t": 350.0, "cap0": 16384, "cap1": 131072, "ok": True, "dt": 1.5},
       {"k": "evict", "t": 355.0, "ok": True}, {"k": "evict", "t": 356.0, "ok": False},
       {"k": "grow", "t": 360.0, "cap0": 16384, "cap1": 16384, "ok": False, "dt": 0.1},
       {"k": "st", "t": 380.0, "waits": 0, "ends": 0, "retries": 0},
       {"k": "st", "t": 399.0, "waits": 1, "ends": 0, "retries": 2},
       {"k": "grow", "t": 450.0, "cap0": 8192, "cap1": 16384, "ok": True, "dt": 0.3}]
G = [{"k": "md", "t": 590.0, "room": 26_120_000_000, "reserve": 2_147_483_648, "window_bytes": 4_800_000_000},
     {"k": "st", "t": 790.0, "waits": 0, "ends": 0, "retries": 0}]


def _tool():
    spec = importlib.util.spec_from_file_location("stress_digest", HERE / "stress_digest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(d: Path) -> Path:
    rows = [{"tag": a, "phase": b, "t": float(t), "memavail_mib": float(m), "free_mib": float(f)}
            for a, b, t, m, f in PROBES]
    (d / "probe.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (d / "diag_off_1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in OFF) + "not json\n", encoding="utf-8")
    (d / "diag_g_1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in G), encoding="utf-8")
    return d


def test_room_and_footprints(tmp_path):
    t = _tool().digest(_write(tmp_path), ["off_1", "g_1"])["tags"]
    off, g = t["off_1"], t["g_1"]
    assert (off["room_gib"], off["reserve_gib"], off["window_gib"], off["full_window_streams"]) == (24.401, 2.0, 4.47, 5)
    assert (g["room_gib"], g["full_window_streams"]) == (24.326, 4)
    assert off["probes"]["pre"] == {"memavail_mib": 60000.0, "free_mib": 59000.0, "foot_avail_mib": 0.0,
                                    "foot_free_mib": 0.0}
    assert off["probes"]["stress"]["foot_avail_mib"] == 26000.0 and off["probes"]["drive"]["foot_free_mib"] == 24900.0
    assert (off["base"], off["vs_base"]) == (None, None)
    assert g["base"] == "off_1"
    assert g["vs_base"] == {"room_gib": -0.075, "idle": {"foot_avail_mib": 600.0, "foot_free_mib": 550.0},
                            "drive": {"foot_avail_mib": 50.0, "foot_free_mib": 50.0},
                            "stress": {"foot_avail_mib": 100.0, "foot_free_mib": 50.0}}


def test_segments_and_counters(tmp_path):
    off = _tool().digest(_write(tmp_path), ["off_1", "g_1"])["tags"]["off_1"]
    seg = off["segments"]
    zero = {"grows": 0, "refused": 0, "slow_grows": 0, "grow_s_max": 0.0, "grow_s_sum": 0.0, "evicts": 0}
    assert seg["idle"] == zero
    assert seg["drive"] == {**zero, "grows": 1, "grow_s_max": 0.2, "grow_s_sum": 0.2}
    assert seg["stress"] == {"grows": 1, "refused": 1, "slow_grows": 1, "grow_s_max": 1.5, "grow_s_sum": 1.6,
                             "evicts": 1}
    assert seg["all"] == {"grows": 3, "refused": 1, "slow_grows": 1, "grow_s_max": 1.5, "grow_s_sum": 2.1,
                          "evicts": 1}
    assert list(seg) == ["idle", "drive", "stress", "all"]
    assert off["last_st"] == {"waits": 1, "ends": 0, "retries": 2}


def test_missing_inputs(tmp_path):
    t = _tool().digest(tmp_path, ["x_2"])["tags"]["x_2"]
    assert (t["room_gib"], t["full_window_streams"], t["probes"], t["segments"], t["last_st"]) == (None, None, {}, {},
                                                                                                   None)
    assert (t["base"], t["vs_base"]) == (None, None)


def test_cli_writes_json_and_markdown(tmp_path, capsys):
    mod = _tool()
    _write(tmp_path)
    rc = mod.main([str(tmp_path), "--tags", "off_1,g_1", "--json", str(tmp_path / "d.json"),
                   "--markdown", str(tmp_path / "d.md")])
    assert rc == 0
    assert json.loads((tmp_path / "d.json").read_text(encoding="utf-8"))["tags"]["g_1"]["base"] == "off_1"
    md = (tmp_path / "d.md").read_text(encoding="utf-8")
    assert md == capsys.readouterr().out
    for s in ("| off_1 |", "| g_1 |", "24.401", "24.326", "-0.075", "26000.0", "+600.0", "| stress |"):
        assert s in md
