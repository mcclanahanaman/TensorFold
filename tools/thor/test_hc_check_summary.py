"""tools/thor/hc_check_summary.py: hc_check.py kernels' per-call rows -> per prompt and row count M, over layers."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import hc_check_summary as hs  # noqa: E402

ROWS = [
    {"prompt": "nat32k", "pass": 0, "layer": 0, "hc": "attn", "mode": 0, "R": 4096, "M": 4096,
     "equal_act": True, "equal_inject": True, "equal_mixed": True,
     "err_act_r25": 2e-3, "err_act_new": 2e-3, "err_mixed_r25": 4e-3, "err_mixed_new": 4e-3,
     "errmax_act_r25": 5e-3, "errmax_act_new": 5e-3, "errmax_mixed_r25": 8e-3, "errmax_mixed_new": 8e-3,
     "ms_wb": 0.5, "ms_r25": 4.0, "ms_r25_down": 2.0, "ms_new": 1.0, "ms_new_down_act": 0.6},
    {"prompt": "nat32k", "pass": 0, "layer": 13, "hc": "mlp", "mode": 1, "R": 4096, "M": 4096,
     "equal_act": True, "equal_inject": None, "equal_mixed": True,
     "err_act_r25": 1e-3, "err_act_new": 1.5e-3, "err_mixed_r25": 2e-3, "err_mixed_new": 2e-3,
     "errmax_act_r25": 2e-3, "errmax_act_new": 2e-3, "errmax_mixed_r25": 4e-3, "errmax_mixed_new": 6e-3,
     "ms_wb": 0.7, "ms_r25": 6.0, "ms_r25_down": 3.0, "ms_new": 2.0, "ms_new_down_act": 1.0},
    {"prompt": "nat32k", "pass": 7, "layer": 13, "hc": "attn", "mode": 2, "R": 4096, "M": 4096,
     "equal_act": False, "equal_inject": True, "equal_mixed": True,
     "err_act_r25": 0.0, "err_act_new": 0.0, "err_mixed_r25": 1e-3, "err_mixed_new": 1e-3,
     "errmax_act_r25": 0.0, "errmax_act_new": 0.0, "errmax_mixed_r25": 1e-3, "errmax_mixed_new": 1e-3,
     "ms_wb": 0.9, "ms_r25": 5.0, "ms_r25_down": "RuntimeError('x')", "ms_new": 4.0, "ms_new_down_act": 0.8},
    {"prompt": "nat32k", "pass": 0, "layer": 0, "hc": "attn", "mode": 0, "R": 4096, "M": 128,
     "equal_act": True, "equal_inject": True, "equal_mixed": True,
     "err_act_r25": 2e-3, "err_act_new": 2e-3, "err_mixed_r25": 4e-3, "err_mixed_new": 4e-3,
     "errmax_act_r25": 5e-3, "errmax_act_new": 5e-3, "errmax_mixed_r25": 8e-3, "errmax_mixed_new": 8e-3},
    {"prompt": "syn128k", "pass": 31, "layer": 0, "hc": "attn", "mode": 2, "R": 4096, "M": 4096,
     "bench_error": "boom"},
]


def test_summarize_groups_by_prompt_and_rows():
    s = hs.summarize(ROWS)
    a = s["nat32k"]["4096"]
    assert a["calls"] == 3
    assert a["equal_act"] is False and a["equal_inject"] is True and a["equal_mixed"] is True   # None ignored
    assert a["err_ratio_act_max"] == 1.5                 # 0/0 counts as 1.0; the largest new/r25 ratio
    assert a["err_ratio_mixed_max"] == 1.0
    assert a["errmax_mixed_new"] == 8e-3 and a["errmax_act_r25"] == 5e-3        # maxima of the per-call maxima
    assert a["ms_r25"] == 5.0 and a["ms_new"] == 2.0 and a["ms_wb"] == 0.7      # medians
    assert a["ms_r25_down"] == 2.5                       # median of the numeric values only
    assert a["speedup"] == 2.5                           # median ms_r25 / median ms_new
    b = s["nat32k"]["128"]
    assert b["calls"] == 1 and b["err_ratio_act_max"] == 1.0 and "ms_r25" not in b and "speedup" not in b
    c = s["syn128k"]["4096"]
    assert c["calls"] == 1 and c["errors"] == ["boom"]
    assert "errors" not in a


def test_markdown_and_main(tmp_path, capsys):
    src = tmp_path / "hc_check.json"
    src.write_text(json.dumps({"upmix": True, "rows": ROWS}))
    out, md = tmp_path / "s.json", tmp_path / "s.md"
    assert hs.main([str(src), "--json", str(out), "--markdown", str(md)]) == 0
    got = json.loads(out.read_text())
    assert got["upmix"] is True
    assert got["summary"]["nat32k"]["4096"]["speedup"] == 2.5
    text = md.read_text()
    assert text.splitlines()[0].startswith("| prompt | M | calls |")
    assert "| nat32k | 4096 | 3 | no | yes | yes | 1.50 | 1.00 | 5.000 | 2.000 | 2.50 |" in text
    assert "| nat32k | 128 | 1 | yes | yes | yes | 1.00 | 1.00 | - | - | - |" in text
    assert "| syn128k | 4096 | 1 | - | - | - | - | - | - | - | - |" in text
    assert text in capsys.readouterr().out
