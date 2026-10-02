"""tools/thor/glue_digest.py (THOR-EXPERT-GLUE): two kernel dumps (off, on) -> the MoE glue's classes per prompt."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import glue_digest as gd  # noqa: E402

W4 = "kernel_cutlass_kernel_flashinferfused_moe_Sm100W4A16GroupedGemmKernel_x"


def _dump(scale: float, scatter: bool) -> dict:
    ks = [(f"{W4}|10x1144x1|512", 392, 1740.0), (f"{W4}|2x1x10|512", 392, 1026.0),
          ("gather|4096x1x1|128", 391, 384.4), ("gather|7x1x1|128", 1, 0.6), ("_router|32x9x1|128", 392, 91.8),
          ("_topk_rows|4096x1x1|128", 392, 61.9), ("plan_rank|44x1x1|1024", 392, 5.0),
          ("plan_offsets|1x1x1|1024", 392, 6.0), ("tiles|1217x1x1|32", 392, 2.4),
          ("_hc_writeback|4096x10x1|64", 805, 966.0), ("vectorized_elementwise_kernel|2560x1x1|128", 2744, 149.6),
          ("_row_attn|256x2x1|128", 1546, 1544.7)]
    if scatter:
        ks.append(("scatter|1217x1x1|128", 392, 722.8))
    else:
        ks += [("_swiglu|4096x5x1|32", 392, 20.0), ("_writeback_rows|4096x10x1|64", 392, 700.0)]
    kern = [{"key": k, "n": n, "ms": ms * scale} for k, n, ms in ks]
    total = sum(r["ms"] for r in kern)
    return {"prompt 32768": {"kernel_ms": total, "wall_ms": total + 250.0, "kernels": kern}}


def test_classes_and_deltas_per_prompt():
    out = gd.digest(_dump(1.0, True), _dump(1.0, False))
    p = out["prompts"]["prompt 32768"]
    c = p["classes"]
    assert c["experts_gemm"]["n_off"] == 784 and c["experts_gemm"]["ms_off"] == pytest.approx(2766.0)
    assert c["gather"]["n_off"] == 392 and c["gather"]["ms_off"] == pytest.approx(385.0)
    assert c["plan"]["ms_off"] == pytest.approx(11.0)
    assert c["scatter"]["ms_off"] == pytest.approx(722.8) and c["scatter"]["n_on"] == 0
    assert c["scatter"]["delta_ms"] == pytest.approx(-722.8)
    assert c["shared_swiglu"]["ms_on"] == pytest.approx(20.0) and c["shared_swiglu"]["ms_off"] == 0.0
    assert c["writeback"]["n_on"] == 805 + 392 and c["writeback"]["ms_on"] == pytest.approx(1666.0)
    assert c["torch_elementwise"]["ms_off"] == pytest.approx(149.6)
    assert c["other"]["ms_off"] == pytest.approx(1544.7)
    assert set(c) == set(gd.CLASSES)


def test_glue_totals_shares_and_unmatched_labels():
    off, on = _dump(1.0, True), _dump(1.0, False)
    off["prompt 131072"] = off["prompt 32768"]
    out = gd.digest(off, on)
    p = out["prompts"]["prompt 32768"]
    glue_off = 91.8 + 61.9 + 11.0 + 2.4 + 385.0 + 722.8
    glue_on = 91.8 + 61.9 + 11.0 + 2.4 + 385.0 + 20.0
    assert p["glue_ms_off"] == pytest.approx(glue_off) and p["glue_ms_on"] == pytest.approx(glue_on)
    assert p["glue_share_off"] == pytest.approx(100.0 * glue_off / p["kernel_ms_off"])
    assert p["kernel_ms_on"] - p["kernel_ms_off"] == pytest.approx(p["kernel_ms_delta"])
    assert p["wall_ms_off"] == pytest.approx(p["kernel_ms_off"] + 250.0)
    assert out["unmatched"] == ["prompt 131072"]


def test_cli_writes_json_and_markdown(tmp_path):
    a, b = tmp_path / "off.json", tmp_path / "on.json"
    a.write_text(json.dumps(_dump(1.0, True)))
    b.write_text(json.dumps(_dump(1.0, False)))
    js, md = tmp_path / "d.json", tmp_path / "d.md"
    subprocess.run([sys.executable, str(Path(gd.__file__)), str(a), str(b), "--json", str(js), "--markdown", str(md)],
                   check=True)
    got = json.loads(js.read_text())
    assert "prompt 32768" in got["prompts"]
    text = md.read_text()
    assert "| class | n off | ms off | n on | ms on | delta ms |" in text
    assert "| scatter | 392 | 722.8 | 0 | 0.0 | -722.8 |" in text
    assert "glue" in text and "prompt 32768" in text
