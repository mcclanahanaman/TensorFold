"""Spec for tools/thor/rows2_digest.py (ROWS-2: per-request memory, n-gram residency, faults and server counters).

    python -m pytest -q tools/thor/test_rows2_digest.py

Inputs per server tag in one folder: marks_<tag>.jsonl (marks_run.py), mw_<tag>.jsonl (memwatch.py) and
diag_<tag>.jsonl (serve_diag.py). Standard library and pytest only; no GPU, no TensorFold import.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rows2_digest as rd  # noqa: E402

GIB_KB = 1048576
GIB = 2**30


def _mark(tool, seq, t0, t1):
    return {"seq": seq, "tool": tool, "t0": t0, "t1": t1, "ok": True, "chars": 1000, "max_tokens": 1}


MARKS = [_mark("natural_prefill", 0, 100.0, 110.0), _mark("natural_prefill", 1, 115.0, 125.0),
         _mark("serve_ab", 0, 130.0, 134.0),
         _mark("serve_ab", 9, 140.0, 150.0), _mark("serve_ab", 10, 140.1, 151.0),
         _mark("serve_ab", 11, 140.2, 149.5), _mark("serve_ab", 12, 140.3, 152.0),
         _mark("serve_ab", 13, 160.0, 161.0)]

VM0 = {"pgmajfault": 1000, "workingset_refault_file": 50, "pgscan_kswapd": 10, "pgscan_direct": 0,
       "pgsteal_kswapd": 10, "pgsteal_direct": 0, "allocstall_normal": 0, "allocstall_movable": 0}
VM1 = {"pgmajfault": 6000, "workingset_refault_file": 1050, "pgscan_kswapd": 2010, "pgscan_direct": 500,
       "pgsteal_kswapd": 1810, "pgsteal_direct": 400, "allocstall_normal": 3, "allocstall_movable": 2}
VM2 = {**VM1, "pgmajfault": 6300, "workingset_refault_file": 1100}
VM3 = {**VM2, "pgmajfault": 6400, "workingset_refault_file": 1180, "pgscan_kswapd": 3010, "pgsteal_kswapd": 2810}
VM4 = {**VM3, "pgmajfault": 6410, "workingset_refault_file": 1190, "allocstall_normal": 4}


def _mw(t, free, avail, cached, majflt, vm, psi, table="absent"):
    rec = {"t": t, "mem": {"MemTotal": 128 * GIB_KB, "MemFree": int(free * GIB_KB),
                           "MemAvailable": int(avail * GIB_KB), "Cached": int(cached * GIB_KB)},
           "vm": dict(vm), "psi": psi, "proc": {"majflt": majflt, "minflt": 7, "utime": 1, "stime": 1}}
    if table != "absent":
        rec["table"] = table
        rec["res_s"] = 0.05
    return rec


PSI0, PSI1, PSI2 = {"some": 1000, "full": 500}, {"some": 251000, "full": 120500}, {"some": 252000, "full": 121000}
PSI3, PSI4 = {"some": 260000, "full": 125000}, {"some": 260500, "full": 125000}
MW = [
    {"t": 90.0, "start": {"pid": 1, "every": 1.0, "table_gib": 26.822, "table_ranges": 3}},
    _mw(99.0, 2.0, 30.0, 28.0, 100, VM0, PSI0, table=1.0),
    _mw(105.0, 1.5, 29.0, 28.0, 3000, VM0, PSI0, table=0.5),
    _mw(111.0, 1.0, 29.5, 28.5, 5100, VM1, PSI1, table=0.05123),
    _mw(114.0, 1.0, 29.5, 28.5, 5100, VM1, PSI1, table=None),
    _mw(126.0, 0.5, 29.0, 28.5, 5300, VM2, PSI2, table=0.98712),
    _mw(129.0, 0.5, 29.0, 28.5, 5300, VM2, None),
    _mw(135.0, 0.25, 28.0, 27.75, 5400, VM3, PSI3, table=0.9),
    _mw(139.0, 0.25, 28.0, 27.75, 5400, VM3, PSI3, table=0.9),
    _mw(153.0, 0.25, 28.0, 27.75, 5410, VM4, PSI4, table=0.875),
    _mw(159.0, 0.25, 28.0, 27.75, 5410, VM4, PSI4, table=0.875),
    _mw(162.0, 0.25, 28.0, 27.75, 5411, VM4, PSI4, table=0.875),
    {"t": 163.0, "end": {"samples": 11}},
]

ST0 = {"g_calls": 0, "g_rows": 0, "g_s": 0.0, "stages": 0, "stage_s": 0.0, "comp_s": 0.0, "passes": 0, "pass_s": 0.0,
       "rounds": 0, "round_s": 0.0, "grows": 0, "grow_s": 0.0, "refused": 0, "evicts": 0, "shrinks": 0, "keeps": 0,
       "ec_calls": 0, "ec_s": 0.0, "majflt": 90, "minflt": 5, "room": 23 * GIB, "reserve": 2 * GIB, "held": 0,
       "waits": 0, "ends": 0, "kept": 0, "kept_rows": 0, "live": 0, "filling": 0, "free_slots": 4,
       "alloc": 59 * GIB, "reserved": 60 * GIB, "reserved_peak": 60 * GIB, "dev_allocs": 100, "dev_frees": 10,
       "retries": 0}
ST1 = {**ST0, "g_calls": 480, "g_rows": 960000, "g_s": 2.5, "stages": 16, "stage_s": 3.0, "comp_s": 0.5, "passes": 16,
       "pass_s": 9.5, "rounds": 1, "round_s": 9.6, "grows": 4, "ec_calls": 4, "ec_s": 0.2, "held": 2 * GIB,
       "kept": 1, "kept_rows": 32768, "reserved": int(61.5 * GIB), "dev_allocs": 108}
ST2 = {**ST1, "g_calls": 960, "g_rows": 1920000, "g_s": 2.75, "stages": 32, "stage_s": 3.25, "comp_s": 1.0,
       "passes": 32, "pass_s": 19.0, "rounds": 2, "round_s": 19.25, "evicts": 1}
ST3 = {**ST2, "g_calls": 1000, "g_rows": 2000000, "g_s": 3.0, "stages": 36, "stage_s": 3.5, "comp_s": 1.25,
       "passes": 36, "pass_s": 23.0, "rounds": 3, "round_s": 23.25, "grows": 5, "ec_calls": 5, "ec_s": 0.25,
       "held": int(2.5 * GIB), "kept": 2, "kept_rows": 40960, "reserved": 62 * GIB, "dev_allocs": 110}
ST4 = {**ST3, "rounds": 103, "round_s": 35.25, "g_calls": 1400, "g_rows": 2001600, "g_s": 3.125, "stages": 136,
       "stage_s": 4.5}
ST5 = {**ST4, "rounds": 104, "round_s": 35.75}


def _st(t, base):
    return {"k": "st", "t": t, **base}


DIAG = [
    {"k": "start", "t": 90.0, "pid": 1, "argv": [], "ballast_gib": 3.0, "rows_env": None},
    {"k": "md", "t": 95.0, "room": 23 * GIB, "reserve": 2 * GIB, "prefill_rows": 2048, "slots": 4,
     "window_bytes": 4800000000, "slot_bytes": 1000},
    _st(99.5, ST0),
    {"k": "pass", "i": 99, "t0": 99.9, "t": 99.95, "dt": 9.0, "rows": [2048], "starts": [0], "plens": [32768],
     "majflt": 777},
    {"k": "pass", "i": 0, "t0": 100.5, "t": 101.0, "dt": 0.5, "rows": [2048], "starts": [0], "plens": [32768],
     "majflt": 100},
    {"k": "pass", "i": 1, "t0": 101.0, "t": 101.75, "dt": 0.75, "rows": [2048], "starts": [2048], "plens": [32768],
     "majflt": 200},
    {"k": "pass_gpu", "t": 101.8, "i": 0, "ms": 400.0},
    {"k": "pass", "i": 2, "t0": 101.75, "t": 102.375, "dt": 0.625, "rows": [2048], "starts": [4096], "plens": [32768],
     "majflt": 0},
    {"k": "pass_gpu", "t": 102.4, "i": 1, "ms": 450.0},
    {"k": "grow", "t": 102.5, "rows": 9000, "cap0": 8192, "cap1": 16384, "ok": True, "alone": True, "dt": 0.1,
     "held": GIB},
    _st(110.5, ST1),
    _st(114.5, ST1),
    {"k": "pass", "i": 3, "t0": 125.0, "t": 125.25, "dt": 0.25, "rows": [512], "starts": [32256],
     "plens": [32768], "majflt": 0},
    {"k": "pass_gpu", "t": 125.3, "i": 3, "ms": 200.0},
    {"k": "st_error", "t": 125.4, "error": "boom"},
    _st(125.5, ST2),
    _st(129.5, ST2),
    _st(134.5, ST3),
    _st(139.5, ST3),
    _st(152.5, ST4),
    _st(159.5, ST4),
    _st(161.5, ST5),
]


def _write(folder: Path, tag: str, marks=MARKS, mw=MW, diag=DIAG) -> None:
    (folder / f"marks_{tag}.jsonl").write_text("".join(json.dumps(m) + "\n" for m in marks))
    (folder / f"mw_{tag}.jsonl").write_text("".join(json.dumps(m) + "\n" for m in mw) + '{"t": 163.5, "mem": {"MemF\n')
    (folder / f"diag_{tag}.jsonl").write_text("".join(json.dumps(m) + "\n" for m in diag) + "not json\n\n")


@pytest.fixture()
def folder(tmp_path: Path) -> Path:
    _write(tmp_path, "x_1")
    _write(tmp_path, "y_1", marks=MARKS[:2])
    return tmp_path


def test_load_jsonl_skips_bad_lines(folder):
    assert len(rd.load_jsonl(folder / "mw_x_1.jsonl")) == len(MW)
    assert len(rd.load_jsonl(folder / "diag_x_1.jsonl")) == len(DIAG)


def test_windows_labels_and_merge():
    wins = rd.windows(MARKS)
    assert [w["label"] for w in wins] == ["nat_cold", "nat_warm", "p8k_0", "d4", "serve_ab_13"]
    assert [w["n"] for w in wins] == [1, 1, 1, 4, 1]
    assert (wins[3]["t0"], wins[3]["t1"]) == (140.0, 152.0)
    assert rd.SERVE_AB[:9] == ["p8k_0", "p8k_1", "p32k_0", "p32k_1", "p128k_0", "p128k_1", "f60k_1", "f60k_2", "d1"]
    assert rd.SERVE_AB[9:] == ["d4"] * 4
    assert rd.NATURAL == ["nat_cold", "nat_warm"]


def test_host_side_rows(folder):
    d = rd.digest_tag(folder, "x_1")
    rows = {r["label"]: r for r in d["requests"]}
    assert list(rows) == ["nat_cold", "nat_warm", "p8k_0", "d4", "serve_ab_13"]
    c = rows["nat_cold"]
    assert (c["n"], c["t0"], c["dur_s"]) == (1, 100.0, 10.0)
    assert (c["free_before"], c["free_after"], c["avail_before"], c["avail_after"]) == (2.0, 1.0, 30.0, 29.5)
    assert (c["cached_before"], c["cached_after"]) == (28.0, 28.5)
    assert (c["table_before"], c["table_after"]) == (100.0, 5.1)
    assert (c["majflt"], c["refault"], c["pgscan"], c["pgsteal"], c["allocstall"]) == (5000, 1000, 2500, 2200, 5)
    assert (c["psi_some_ms"], c["psi_full_ms"]) == (250.0, 120.0)
    w = rows["nat_warm"]
    assert (w["free_before"], w["free_after"], w["avail_after"]) == (1.0, 0.5, 29.0)
    assert (w["table_before"], w["table_after"]) == (5.1, 98.7)     # 114.0's residency is null: 111.0's is used
    assert (w["majflt"], w["refault"], w["pgscan"], w["pgsteal"], w["allocstall"]) == (200, 50, 0, 0, 0)
    assert (w["psi_some_ms"], w["psi_full_ms"]) == (1.0, 0.5)
    p = rows["p8k_0"]
    assert (p["free_before"], p["free_after"], p["cached_after"]) == (0.5, 0.25, 27.75)
    assert (p["table_before"], p["table_after"]) == (98.7, 90.0)     # 129.0 has no residency: 126.0's is used
    assert (p["majflt"], p["refault"], p["pgscan"], p["pgsteal"]) == (100, 80, 1000, 1000)
    assert (p["psi_some_ms"], p["psi_full_ms"]) == (None, None)       # 129.0's psi is null
    q = rows["d4"]
    assert (q["n"], q["dur_s"], q["majflt"], q["refault"], q["allocstall"]) == (4, 12.0, 10, 10, 1)
    assert (q["table_before"], q["table_after"], q["psi_some_ms"], q["psi_full_ms"]) == (90.0, 87.5, 0.5, 0.0)
    assert rows["serve_ab_13"]["majflt"] == 1


def test_server_side_rows(folder):
    d = rd.digest_tag(folder, "x_1")
    assert (d["tag"], d["ballast_gib"], d["prefill_rows"], d["room_gib"], d["reserve_gib"]) == ("x_1", 3.0, 2048,
                                                                                                23.0, 2.0)
    rows = {r["label"]: r for r in d["requests"]}
    c = rows["nat_cold"]
    assert (c["g_s"], c["g_rows"], c["stage_s"], c["comp_s"], c["pass_s"], c["passes"]) == (2.5, 960000, 3.0, 0.5,
                                                                                            9.5, 16)
    assert (c["rounds"], c["round_s"], c["grows"], c["evicts"], c["ec_calls"], c["ec_s"]) == (1, 9.6, 4, 0, 4, 0.2)
    assert (c["dev_allocs"], c["held_gib"], c["kept"], c["kept_rows"], c["reserved_gib"]) == (8, 2.0, 1, 32768, 61.5)
    assert (c["pass_n"], c["pass_max_s"], c["pass_majflt"], c["gpu_s"]) == (3, 0.75, 300, 0.85)
    w = rows["nat_warm"]
    assert (w["g_s"], w["g_rows"], w["stage_s"], w["comp_s"], w["pass_s"], w["passes"]) == (0.25, 960000, 0.25, 0.5,
                                                                                            9.5, 16)
    assert (w["round_s"], w["evicts"], w["ec_calls"], w["ec_s"], w["dev_allocs"]) == (9.65, 1, 0, 0.0, 0)
    assert (w["pass_n"], w["pass_max_s"], w["pass_majflt"], w["gpu_s"]) == (1, 0.25, 0, 0.2)   # t0 == t1 counts
    p = rows["p8k_0"]
    assert (p["g_s"], p["g_rows"], p["pass_s"], p["passes"], p["ec_calls"], p["ec_s"]) == (0.25, 80000, 4.0, 4, 1, 0.05)
    assert (p["held_gib"], p["kept"], p["kept_rows"], p["reserved_gib"], p["dev_allocs"]) == (2.5, 2, 40960, 62.0, 2)
    assert (p["pass_n"], p["pass_max_s"], p["pass_majflt"], p["gpu_s"]) == (0, None, 0, None)
    q = rows["d4"]
    assert (q["rounds"], q["round_s"], q["g_s"], q["g_rows"], q["stage_s"], q["comp_s"], q["passes"]) == (
        100, 12.0, 0.125, 1600, 1.0, 0.0, 0)
    assert (rows["serve_ab_13"]["rounds"], rows["serve_ab_13"]["round_s"]) == (1, 0.5)


def test_row_keys_in_order(folder):
    d = rd.digest_tag(folder, "x_1")
    assert list(d["requests"][0]) == rd.ROW_KEYS
    assert rd.ROW_KEYS[:4] == ["label", "n", "t0", "dur_s"]
    assert list(d) == ["tag", "ballast_gib", "prefill_rows", "room_gib", "reserve_gib", "requests"]


def test_compare_and_markdown(folder):
    ds = [rd.digest_tag(folder, "x_1"), rd.digest_tag(folder, "y_1")]
    cmp = rd.compare(ds, ["dur_s", "majflt"])
    assert [c["label"] for c in cmp] == ["nat_cold", "nat_warm", "p8k_0", "d4", "serve_ab_13"]
    assert cmp[0]["dur_s"] == {"x_1": 10.0, "y_1": 10.0}
    assert cmp[2]["majflt"] == {"x_1": 100, "y_1": None}
    md = rd.markdown(ds, ["dur_s", "majflt"])
    assert "## dur_s\n\n| label | x_1 | y_1 |\n|---|---|---|\n| nat_cold | 10.0 | 10.0 |\n" in md
    assert "| p8k_0 | 100 | - |" in md
    assert "## x_1 (ballast 3.0 GiB, prefill rows 2048, room 23.0 GiB, reserve 2.0 GiB)" in md
    assert "| label | " + " | ".join(rd.DETAIL) + " |" in md
    assert rd.DETAIL[:4] == ["dur_s", "table_before", "table_after", "majflt"]


def test_main_writes_outputs(folder, capsys):
    rc = rd.main([str(folder), "--tags", "x_1,y_1", "--json", str(folder / "d.json"), "--markdown",
                  str(folder / "d.md")])
    assert rc == 0
    got = json.loads((folder / "d.json").read_text())
    assert [d["tag"] for d in got["digests"]] == ["x_1", "y_1"]
    assert got["compare"][0]["label"] == "nat_cold" and "dur_s" in got["compare"][0]
    assert "## dur_s" in (folder / "d.md").read_text()
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 2 and out[0].startswith("x_1: 5 requests")


def test_main_missing_tag(folder, capsys):
    assert rd.main([str(folder), "--tags", "x_1,zz_9"]) == 1
    assert "zz_9" in capsys.readouterr().err
