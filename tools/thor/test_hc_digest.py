"""tools/thor/hc_digest.py: two kernel dumps (path off and on) -> the hyper-connection kernels per prompt, side by
side, with the family's total and share of kernel time."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import hc_digest as hd  # noqa: E402

OFF = {"prompt 32768": {"kernel_ms": 1200.0, "wall_ms": 1300.0, "kernels": [
    {"key": "_hc_writeback|4096x10x1|64", "n": 96, "ms": 100.0},
    {"key": "_hc_normed|4096x20x1|128", "n": 96, "ms": 60.0},
    {"key": "_b16mm|32x6x32|128", "n": 96, "ms": 120.0},
    {"key": "_b16mm|1x6x32|128", "n": 2, "ms": 1.0},
    {"key": "_reduce|1296x1x1|128", "n": 96, "ms": 40.0},
    {"key": "_hc_act|4096x1x1|128", "n": 96, "ms": 10.0},
    {"key": "_hc_mix|4096x10x1|64", "n": 96, "ms": 70.0},
    {"key": "nvjet_x|10x2x1|384", "n": 500, "ms": 400.0}]}}
ON = {"prompt 32768": {"kernel_ms": 1000.0, "wall_ms": 1080.0, "kernels": [
    {"key": "_hc_writeback|4096x10x1|64", "n": 96, "ms": 100.0},
    {"key": "_down_act|192x1x1|128", "n": 96, "ms": 50.0},
    {"key": "_mix_h|4096x10x1|64", "n": 96, "ms": 72.0},
    {"key": "nvjet_x|10x2x1|384", "n": 500, "ms": 400.0}]},
    "prompt 131072": {"kernel_ms": 10.0, "wall_ms": 12.0, "kernels": []}}


def test_ops_sum_launches_and_ms_by_name_over_grids():
    ops = hd.ops(OFF["prompt 32768"])
    assert ops["down (b16 split-K)"] == {"n": 98, "ms": 121.0, "grids": 2}
    assert ops["writeback"] == {"n": 96, "ms": 100.0, "grids": 1}
    assert "nvjet_x" not in ops and len(ops) == 6
    on = hd.ops(ON["prompt 32768"])
    assert set(on) == {"writeback", "down + act (fused)", "mix from h"}


def test_digest_compares_off_and_on_per_prompt():
    d = hd.digest(OFF, ON)
    p = d["prompt 32768"]
    assert p["off"]["family_ms"] == 401.0 and p["on"]["family_ms"] == 222.0
    assert abs(p["off"]["family_share"] - 401.0 / 1200.0) < 1e-12 and abs(p["on"]["family_share"] - 0.222) < 1e-12
    assert p["saved_ms"] == 179.0 and p["kernel_saved_ms"] == 200.0
    assert d["prompt 131072"]["off"] is None and d["prompt 131072"]["on"]["family_ms"] == 0.0


def test_main_writes_json_and_markdown(tmp_path, capsys):
    a, b = tmp_path / "off.json", tmp_path / "on.json"
    a.write_text(json.dumps(OFF))
    b.write_text(json.dumps(ON))
    out, md = tmp_path / "d.json", tmp_path / "d.md"
    assert hd.main([str(a), str(b), "--json", str(out), "--markdown", str(md)]) == 0
    assert json.loads(out.read_text())["prompt 32768"]["saved_ms"] == 179.0
    text = md.read_text()
    assert "| prompt 32768 | down (b16 split-K) | 98 | 121.0 | 10.1% | 0 | 0.0 | 0.0% |" in text
    assert "| prompt 32768 | down + act (fused) | 0 | 0.0 | 0.0% | 96 | 50.0 | 5.0% |" in text
    assert "| prompt 32768 | hc family | 578 | 401.0 | 33.4% | 288 | 222.0 | 22.2% |" in text
    assert "| prompt 32768 | up + mix (fused) |" not in text          # no launches on either side: no row
    assert "| prompt 131072 | hc family | - | - | - | 0 | 0.0 | 0.0% |" in text
    assert text in capsys.readouterr().out
