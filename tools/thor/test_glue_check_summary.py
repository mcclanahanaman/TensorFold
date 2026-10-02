"""tools/thor/glue_check_summary.py (THOR-EXPERT-GLUE): glue_check.json -> byte-equality counts, fp64 error ratios
and per-call timings."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import glue_check_summary as gs  # noqa: E402

FLAGS = ("equal_h", "equal_pss", "equal_act", "equal_shared")


def _row(M: int, R: int = 4096, **kw) -> dict:
    r = {"prompt": "nat32k", "pass": 0, "layer": 0, "R": R, "M": M, **{f: True for f in FLAGS},
         "err_h_off_mean": 1e-3, "err_h_off_max": 2e-3, "err_h_on_mean": 1e-3, "err_h_on_max": 2e-3,
         "err_act_off_mean": 2e-3, "err_act_off_max": 4e-3, "err_act_on_mean": 2e-3, "err_act_on_max": 4e-3}
    if M == R:
        r.update({"equal_served": True, "ms_moe_off": 10.0, "ms_moe_on": 8.0, "ms_wb_off": 1.6, "ms_wb_on": 1.7,
                  "ms_shared_off": 1.2, "ms_shared_on": 0.4, "ms_scatter": 1.8})
    r.update(kw)
    return r


def _check(rows: list) -> dict:
    return {"note": "n", "active": True, "layers": [0, 16, 32, 47], "rows": rows}


def test_equality_counts_and_errors():
    rows = [_row(128), _row(4096), _row(4096, layer=16, equal_pss=False, equal_served=None),
            {"prompt": "syn128k", "pass": 3, "layer": 0, "R": 4096, "M": 4096, "bench_error": "OOM"}]
    out = gs.summarize(_check(rows))
    assert out["rows"] == 3 and out["errors"] == 1
    assert out["equal"]["equal_h"] == [3, 3] and out["equal"]["equal_pss"] == [2, 3]
    assert out["equal"]["equal_served"] == [1, 1]                       # None is not counted
    assert out["all_equal"] is False
    ok = gs.summarize(_check([_row(128), _row(4096)]))
    assert ok["all_equal"] is True and ok["errors"] == 0


def test_error_ratio_and_by_m():
    rows = [_row(128), _row(512, err_h_on_max=3e-3), _row(4096, err_act_on_max=6e-3), _row(4096, err_h_off_max=0.0,
                                                                                          err_h_on_max=0.0)]
    out = gs.summarize(_check(rows))
    assert out["err_ratio_max"] == pytest.approx(1.5)                     # 3e-3/2e-3 and 6e-3/4e-3; 0/0 skipped
    assert set(out["by_M"]) == {"128", "512", "4096"}
    b = out["by_M"]["4096"]
    assert b["rows"] == 2 and b["all_equal"] is True
    assert b["err_act_on_max"] == pytest.approx(6e-3) and b["err_h_off_max"] == pytest.approx(2e-3)


def test_timing_medians_and_cli(tmp_path):
    rows = [_row(4096), _row(4096, ms_moe_on=7.0, ms_wb_on=1.9), _row(4096, ms_moe_on=9.0, ms_wb_on=1.5), _row(128)]
    out = gs.summarize(_check(rows))
    t = out["timing"]
    assert t["ms_moe_off"] == pytest.approx(10.0) and t["ms_moe_on"] == pytest.approx(8.0)
    assert t["ms_wb_on"] == pytest.approx(1.7) and t["ms_scatter"] == pytest.approx(1.8)
    assert out["saved_ms_per_call"] == pytest.approx(1.9)               # per row 1.9, 2.7, 1.1: the median
    p = tmp_path / "c.json"
    p.write_text(json.dumps(_check(rows)))
    js, md = tmp_path / "s.json", tmp_path / "s.md"
    subprocess.run([sys.executable, str(Path(gs.__file__)), str(p), "--json", str(js), "--markdown", str(md)],
                   check=True)
    assert json.loads(js.read_text())["rows"] == 4
    text = md.read_text()
    assert "| M | rows | all equal | err h off max | err h on max | err act off max | err act on max |" in text
    assert "| ms_moe_off | 10.000 |" in text
