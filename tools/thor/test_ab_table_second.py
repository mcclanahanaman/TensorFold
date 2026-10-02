"""Spec for tools/thor/ab_table.py's stall-watch row and three-config tables (THOR-K4-QSA-SEL; T-034's lesson: the
second long request on the same server is where a grow stall shows).

    python -m pytest -q tools/thor/test_ab_table_second.py

Standard library and pytest only; no GPU, no TensorFold import.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_table  # noqa: E402


def _pre(rows):
    return {"prefill": [{"target": t, "round": i, "tok_s": v, "token_sha": f"{t}-{i}"} for t, i, v in rows]}


def test_the_second_request_of_a_target_is_its_own_metric():
    """'prefill <k>k second' is the target's second request in report order, right after 'prefill <k>k first'; a
    target sent once has no such row. The mean and the first are unchanged."""

    m = ab_table.run_metrics(_pre([(32768, 0, 2300.0), (131072, 0, 1800.0), (32768, 1, 2310.0),
                                   (131072, 1, 1700.0)]), None)
    keys = list(m)
    assert m["prefill 128k first"] == 1800.0 and m["prefill 128k second"] == 1700.0
    assert m["prefill 128k"] == pytest.approx(1750.0)
    assert m["prefill 32k second"] == 2310.0
    assert keys.index("prefill 128k second") == keys.index("prefill 128k first") + 1
    assert keys.index("prefill 32k second") == keys.index("prefill 32k first") + 1
    once = ab_table.run_metrics(_pre([(8192, 0, 2000.0)]), None)
    assert "prefill 8k second" not in once and once["prefill 8k first"] == 2000.0


def test_three_configs_against_one_baseline(tmp_path):
    """r22 (both switches off), attn (THOR-K4-QSA-ATTN only) and sel (ATTN + SEL) in one table: a ratio column per
    non-baseline config, the second 128k request included."""

    runs = {"r22": (1800.0, 1600.0), "attn": (1850.0, 1700.0), "sel": (1980.0, 1960.0)}
    for cfg, (a, b) in runs.items():
        (tmp_path / f"ab_{cfg}_1.json").write_text(json.dumps(_pre([(131072, 0, a), (131072, 1, b)])))
    md, js = tmp_path / "t.md", tmp_path / "t.json"
    assert ab_table.main([str(tmp_path), "--configs", "r22,attn,sel", "--rounds", "1", "--baseline", "r22",
                          "--json", str(js), "--markdown", str(md)]) == 0
    s = json.loads(js.read_text())["summary"]
    assert s["prefill 128k second"]["ratio"]["sel"] == pytest.approx(1960.0 / 1600.0)
    assert s["prefill 128k second"]["ratio"]["attn"] == pytest.approx(1700.0 / 1600.0)
    lines = md.read_text().splitlines()
    assert lines[0] == "| metric | r22 r1 | attn r1 | sel r1 | attn/r22 | sel/r22 |"
    assert "| prefill 128k second | 1600 | 1700 | 1960 | 1.062 | 1.225 |" in lines
