"""tools/thor/qsa_kernels_summary.py: qsa_check.py kernels' per-slice results -> per prompt and first row, over layers."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import qsa_kernels_summary as ks  # noqa: E402

ROWS = [
    {"prompt": "nat32k", "layer": "3", "p0": 8192, "rows": 256, "attn_thor_ms": 1.0, "attn_new_ms": 0.5,
     "select_tf_ms": 0.3, "select_new_ms": 0.1, "err_ratio_new_max": 1.1, "err_new_mean": 1e-3,
     "select_new_equal": True, "attn_new_equal_64_4_2": None, "attn_new_ms_32_4_3": 0.4},
    {"prompt": "nat32k", "layer": "7", "p0": 8192, "rows": 256, "attn_thor_ms": 3.0, "attn_new_ms": 1.0,
     "select_tf_ms": 0.5, "select_new_ms": 0.2, "err_ratio_new_max": 1.3, "err_new_mean": 3e-3,
     "select_new_equal": True, "attn_new_equal_64_4_2": True, "attn_new_ms_32_4_3": "RuntimeError('x')"},
    {"prompt": "nat32k", "layer": "11", "p0": 8192, "rows": 256, "attn_thor_ms": 2.0, "attn_new_ms": 0.8,
     "select_tf_ms": 0.4, "select_new_ms": 0.15, "err_ratio_new_max": 0.9, "err_new_mean": 2e-3,
     "select_new_equal": False, "attn_new_equal_64_4_2": True, "attn_new_ms_32_4_3": 0.6},
    {"prompt": "syn128k", "layer": "3", "p0": 65280, "rows": 256, "attn_thor_ms": 4.0, "attn_new_ms": 2.0,
     "rows_error": "boom"},
]


def test_summarize_takes_medians_maxima_means_and_all():
    s = ks.summarize(ROWS)
    a = s["nat32k"]["8192"]
    assert a["layers"] == 3
    assert a["attn_thor_ms"] == 2.0 and a["attn_new_ms"] == 0.8 and a["select_new_ms"] == 0.15
    assert a["attn_new_ms_32_4_3"] == 0.5                     # median of the numeric values only
    assert a["err_ratio_new_max"] == 1.3
    assert abs(a["err_new_mean"] - 2e-3) < 1e-15
    assert a["select_new_equal"] is False and a["attn_new_equal_64_4_2"] is True
    b = s["syn128k"]["65280"]
    assert b["layers"] == 1 and b["attn_new_ms"] == 2.0 and b["errors"] == ["rows_error"]
    assert "errors" not in a


def test_speedups_and_main(tmp_path, capsys):
    s = ks.summarize(ROWS)
    sp = ks.speedups(s, [("attn_thor_ms", "attn_new_ms"), ("select_tf_ms", "select_new_ms")])
    assert sp["nat32k"]["8192"]["attn_thor_ms/attn_new_ms"] == 2.5
    assert abs(sp["nat32k"]["8192"]["select_tf_ms/select_new_ms"] - 0.4 / 0.15) < 1e-12
    assert "select_tf_ms/select_new_ms" not in sp["syn128k"]["65280"]
    src = tmp_path / "k.json"
    src.write_text(json.dumps(ROWS))
    out = tmp_path / "o.json"
    assert ks.main([str(src), "--json", str(out)]) == 0
    got = json.loads(out.read_text())
    assert got["summary"]["nat32k"]["8192"]["attn_new_ms"] == 0.8
    assert got["speedups"]["nat32k"]["8192"]["attn_thor_ms/attn_new_ms"] == 2.5
    text = capsys.readouterr().out
    assert "| nat32k | 8192 |" in text
