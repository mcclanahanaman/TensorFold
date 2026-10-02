"""Spec for the memory and G-B rows of tools/thor/ab_table.py: serve_digest.py's mem_<config>_<round>.json and
thor_divergence.py compare's gbc_<config>_<round>.json (each config's probes against the baseline's).

    python -m pytest -q tools/thor/test_ab_table_mem.py tools/thor/test_ab_table.py

Standard library and pytest only; no GPU, no TensorFold import.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_table  # noqa: E402

MEM = {
    "r2048": {"startup_estimate_gib": 87.86, "budget_gib": 106.84, "window": 262144, "workspace_rows": None,
              "workspace_gib": 0.0, "planned_peak_gib": 87.86, "stream_window": 262144, "kv_room_gib": 25.4,
              "full_window_gib": 4.47, "pass_rows": 2048, "reserve_gib": 2.0, "usable_kv_gib": 23.4,
              "full_window_streams": 5},
    "r8192": {"startup_estimate_gib": 87.86, "budget_gib": 106.84, "window": 262144, "workspace_rows": 8192,
              "workspace_gib": 3.35, "planned_peak_gib": 91.2, "stream_window": 262144, "kv_room_gib": 21.9,
              "full_window_gib": 4.47, "pass_rows": 8192, "reserve_gib": 3.35, "usable_kv_gib": 18.55,
              "full_window_streams": 4},
}
MEM_ROWS = ["pass rows", "kv room GiB", "usable kv GiB", "full-window streams", "window tokens",
            "startup estimate GiB", "planned peak GiB", "workspace GiB"]
GBC = [{"name": f"p{i}", "equal": i != 3, "tokens": [300, 300]} for i in range(10)]
GBC[3].update(index=17, a={"token": "x", "top2_margin": 0.06}, b={"token": "y", "top2_margin": 0.02})


def _ab(tok_s: float) -> dict:
    return {"prefill": [{"target": 32768, "round": 0, "tok_s": tok_s, "token_sha": "s1"}], "followups": [],
            "decode": {"one": {"server_decode_tok_s": 38.0, "token_sha": "d1"}}}


@pytest.fixture
def out(tmp_path):
    for c, tok_s in (("r2048", 2200.0), ("r8192", 2400.0)):
        (tmp_path / f"ab_{c}_1.json").write_text(json.dumps(_ab(tok_s)))
        (tmp_path / f"mem_{c}_1.json").write_text(json.dumps(MEM[c]))
    (tmp_path / "gbc_r8192_1.json").write_text(json.dumps(GBC))
    return tmp_path


def test_memory_rows_in_order_and_absent_values_skipped():
    m = ab_table.run_metrics(None, None, MEM["r8192"])
    assert list(m) == MEM_ROWS
    assert m["pass rows"] == 8192 and m["kv room GiB"] == 21.9 and m["usable kv GiB"] == 18.55
    assert m["full-window streams"] == 4 and m["window tokens"] == 262144
    assert m["startup estimate GiB"] == 87.86 and m["planned peak GiB"] == 91.2 and m["workspace GiB"] == 3.35
    sparse = ab_table.run_metrics(None, None, {"pass_rows": 2048, "kv_room_gib": None})
    assert sparse == {"pass rows": 2048}


def test_gb_rows_count_the_equal_probes():
    m = ab_table.run_metrics(None, None, None, GBC)
    assert m == {"G-B probes equal": 9, "G-B probes": 10}
    assert ab_table.run_metrics(None, None, None, []) == {"G-B probes equal": 0, "G-B probes": 0}


def test_two_argument_calls_are_unchanged():
    ab = _ab(2200.0)
    assert ab_table.run_metrics(ab, None) == ab_table.run_metrics(ab, None, None, None)
    assert "kv room GiB" not in ab_table.run_metrics(ab, None)


def test_rows_follow_the_serve_metrics(out):
    assert ab_table.main([str(out), "--configs", "r2048,r8192", "--rounds", "1", "--baseline", "r2048",
                          "--json", str(out / "t.json")]) == 0
    summary = json.loads((out / "t.json").read_text())["summary"]
    keys = list(summary)
    assert keys.index("prefill 32k") < keys.index("decode 1 tok/s") < keys.index("pass rows")
    assert keys[keys.index("pass rows"):] == MEM_ROWS + ["G-B probes equal", "G-B probes"]
    assert summary["kv room GiB"]["mean"] == {"r2048": 25.4, "r8192": 21.9}
    assert summary["kv room GiB"]["ratio"]["r8192"] == pytest.approx(21.9 / 25.4)
    assert summary["G-B probes equal"]["by_run"] == {"r2048 r1": None, "r8192 r1": 9}


def test_markdown_cells_for_memory_and_counts(out, capsys):
    assert ab_table.main([str(out), "--configs", "r2048,r8192", "--rounds", "1", "--baseline", "r2048"]) == 0
    text = capsys.readouterr().out
    assert "| kv room GiB | 25.40 | 21.90 | 0.862 |" in text
    assert "| usable kv GiB | 23.40 | 18.55 | 0.793 |" in text
    assert "| pass rows | 2048 | 8192 | 4.000 |" in text
    assert "| full-window streams | 5 | 4 | 0.800 |" in text
    assert "| window tokens | 262144 | 262144 | 1.000 |" in text
    assert "| G-B probes equal | - | 9 | - |" in text
    assert "| prefill 32k | 2200 | 2400 | 1.091 |" in text          # the existing cells are unchanged


def test_ledger_carries_the_kv_room(out):
    led = out / "ledger.json"
    assert ab_table.main([str(out), "--configs", "r2048,r8192", "--rounds", "1", "--baseline", "r2048",
                          "--ledger", str(led), "--ledger-run", "r8192:1"]) == 0
    got = json.loads(led.read_text())
    assert got["kv_gib"] == 21.9 and got["prefill_32k"] == 2400.0 and got["decode_1"] == 38.0


def test_missing_memory_and_gb_files(tmp_path, capsys):
    (tmp_path / "ab_a_1.json").write_text(json.dumps(_ab(1000.0)))
    assert ab_table.main([str(tmp_path), "--configs", "a,b", "--rounds", "1", "--baseline", "a",
                          "--json", str(tmp_path / "t.json")]) == 0
    summary = json.loads((tmp_path / "t.json").read_text())["summary"]
    assert "kv room GiB" not in summary and "G-B probes" not in summary
