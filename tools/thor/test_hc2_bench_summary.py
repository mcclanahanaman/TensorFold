"""Tests for tools/thor/hc2_bench_summary.py (THOR-HC-FUSE-2): per-kernel timings of the hyper-connection read-out
kernels, a baseline path against a new one, with bytes and GB/s derived from a fixed bytes model."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import hc2_bench_summary as hs  # noqa: E402

TOOL = Path(__file__).parent / "hc2_bench_summary.py"


def test_bytes_model_counts_every_tensor_read_or_written_once() -> None:
    # down: h + W (low + streams rows) + act and inject gates out + the write-back's partial sums
    assert hs.bytes_model("down", 4096) == 83_886_080 + 6_635_520 + 2_654_208 + 655_360
    assert hs.bytes_model("down", 128) == 2_621_440 + 6_635_520 + 82_944 + 20_480
    # upmix: h + act + W + mixed out + partial sums
    assert hs.bytes_model("upmix", 4096) == 83_886_080 + 2_621_440 + 6_553_600 + 20_971_520 + 655_360
    assert hs.bytes_model("upmix", 512, S=4, D=512, LOW=64, NC=2) == (2 * 512 * 4 * 512 + 2 * 512 * 64
                                                                    + 2 * 4 * 512 * 64 + 2 * 512 * 512
                                                                    + 4 * 512 * 2 * 4)
    with pytest.raises(ValueError):
        hs.bytes_model("mix", 128)


def _bench():
    return {"meta": {"run": "hcf2-b1"}, "rows": [
        {"kernel": "down", "path": "r31", "M": 4096, "ms": 1.300, "equal": True},
        {"kernel": "down", "path": "r31", "M": 4096, "ms": 1.285, "equal": True},
        {"kernel": "down", "path": "ws", "M": 4096, "ms": 0.700, "equal": True},
        {"kernel": "down", "path": "ws", "M": 4096, "ms": 0.650, "equal": False},
        {"kernel": "upmix", "path": "ws", "M": 128, "ms": 0.050, "equal": True},
        {"kernel": "upmix", "path": "r31", "M": 128, "ms": 0.040, "equal": True},
        {"kernel": "down", "path": "r31", "M": 128, "ms": 0.100, "equal": True},
        {"kernel": "upmix", "path": "r31", "M": 4096, "ms": 0.912, "equal": True},
    ]}


def test_summarize_keeps_the_best_time_and_any_mismatch() -> None:
    rows = hs.summarize(_bench(), base="r31", new="ws")
    assert [(r["kernel"], r["M"]) for r in rows] == [("down", 128), ("down", 4096), ("upmix", 128), ("upmix", 4096)]
    d = rows[1]
    assert d["bytes"] == hs.bytes_model("down", 4096)
    assert d["ms_base"] == 1.285 and d["ms_new"] == 0.650
    assert d["speedup"] == round(1.285 / 0.650, 3)
    assert d["gbs_base"] == round(hs.bytes_model("down", 4096) / 1.285e6, 1)
    assert d["gbs_new"] == round(hs.bytes_model("down", 4096) / 0.650e6, 1)
    assert d["equal_new"] is False and d["equal_base"] is True
    u = rows[2]
    assert u["speedup"] == round(0.040 / 0.050, 3) and u["equal_new"] is True


def test_failed_measurements_are_skipped() -> None:
    bench = _bench()
    bench["rows"].append({"kernel": "down", "path": "ws", "M": 128, "error": "Traceback ..."})
    assert hs.summarize(bench, base="r31", new="ws") == hs.summarize(_bench(), base="r31", new="ws")


def test_a_missing_path_leaves_its_columns_empty() -> None:
    rows = hs.summarize(_bench(), base="r31", new="ws")
    d128 = rows[0]
    assert d128["ms_new"] is None and d128["speedup"] is None and d128["gbs_new"] is None
    assert d128["equal_new"] is None
    assert d128["ms_base"] == 0.100 and d128["gbs_base"] == round(hs.bytes_model("down", 128) / 0.100e6, 1)


def test_markdown_lists_one_line_a_kernel_and_size() -> None:
    md = hs.markdown(hs.summarize(_bench(), base="r31", new="ws"), base="r31", new="ws")
    lines = md.strip().splitlines()
    assert lines[0] == "| kernel | M | r31 ms | ws ms | speedup | r31 GB/s | ws GB/s | equal |"
    assert lines[1].startswith("|---")
    assert len(lines) == 2 + 4
    assert lines[3] == (f"| down | 4096 | 1.285 | 0.650 | {1.285 / 0.650:.3f} | "
                        f"{hs.bytes_model('down', 4096) / 1.285e6:.1f} | {hs.bytes_model('down', 4096) / 0.650e6:.1f} "
                        f"| no |")
    assert lines[2] == f"| down | 128 | 0.100 | - | - | {hs.bytes_model('down', 128) / 0.100e6:.1f} | - | - |"


def test_the_command_line_writes_json_and_markdown(tmp_path: Path) -> None:
    src = tmp_path / "bench.json"
    src.write_text(json.dumps(_bench()))
    out_json, out_md = tmp_path / "s.json", tmp_path / "s.md"
    done = subprocess.run([sys.executable, str(TOOL), str(src), "--base", "r31", "--new", "ws", "--json",
                           str(out_json), "--markdown", str(out_md)], capture_output=True, text=True, check=True)
    assert done.stdout.splitlines()[0].startswith("| kernel | M |")
    got = json.loads(out_json.read_text())
    assert got["meta"] == {"run": "hcf2-b1"} and len(got["rows"]) == 4
    assert out_md.read_text().strip() == done.stdout.strip()
