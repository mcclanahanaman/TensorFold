"""Spec for tools/thor/ab_table.py (the A/B result tables for serve_ab.py and natural_prefill.py outputs).

    python -m pytest -q tools/thor/test_ab_table.py

Standard library and pytest only; no GPU, no TensorFold import.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_table  # noqa: E402


def _ab(pre: dict, follow: list, one: tuple, many: list, wall: float, shas: list) -> dict:
    """A serve_ab.py report: ``pre`` {target: [tok_s, ...]}, ``follow`` [(target, ttft, prompt, cached)], ``one``
    (tok_s, acceptance), ``many`` [(tok_s, acceptance)], ``shas`` the reply hashes in report order."""

    it = iter(shas)
    prefill = [{"target": t, "round": i, "tok_s": v, "token_sha": next(it)} for t, vs in pre.items()
               for i, v in enumerate(vs)]
    followups = [{"target": t, "first": {"token_sha": next(it)},
                  "next": {"ttft_s": s, "prompt_tokens": p, "cached": c}} for t, s, p, c in follow]
    decode = {"one": {"server_decode_tok_s": one[0], "acceptance": one[1], "token_sha": next(it)},
              "many": {"wall_tok_s": wall, "server_sum_tok_s": sum(v for v, _ in many),
                       "streams": [{"server_decode_tok_s": v, "acceptance": a, "token_sha": next(it)}
                                   for v, a in many]}}
    return {"prefill": prefill, "followups": followups, "decode": decode}


def _nat(cold: list, warm: list, tokens: int = 32768) -> dict:
    return {"tokens": tokens, "runs": [{"kind": "cold", "tok_s": v} for v in cold] +
            [{"kind": "warm", "tok_s": v} for v in warm]}


@pytest.fixture
def out(tmp_path: Path) -> Path:
    shas = [f"h{i}" for i in range(8)]
    runs = {
        ("stock", 1): (_ab({8192: [1000.0, 1100.0]}, [(16384, 0.20, 16500, 16440), (16384, 0.30, 16600, 16500),
                                                       (61440, 0.15, 61600, 61540)],
                           (40.0, 0.60), [(30.0, 0.5), (31.0, None)], 75.0, shas), _nat([900.0], [1000.0, 1200.0])),
        ("stock", 2): (_ab({8192: [1200.0, 1300.0]}, [(16384, 0.22, 16500, 16440), (16384, 0.28, 16600, 16500),
                                                       (61440, 0.17, 61600, 61540)],
                           (42.0, 0.62), [(32.0, 0.7), (33.0, 0.9)], 80.0, shas), None),
        ("on", 1): (_ab({8192: [2000.0, 2200.0]}, [(16384, 0.40, 16500, 16384), (16384, 0.50, 16600, 16448),
                                                    (61440, 0.14, 61600, 61540)],
                        (41.0, 0.64), [(30.0, 0.6), (30.0, 0.8)], 76.0, shas[:4] + ["x"] + shas[5:]),
                     _nat([1800.0], [2000.0])),
        ("on", 2): (_ab({8192: [2400.0, 2600.0]}, [(16384, 0.42, 16500, 16384), (16384, 0.48, 16600, 16448),
                                                    (61440, 0.16, 61600, 61540)],
                        (43.0, 0.66), [(34.0, 0.6), (34.0, 0.6)], 84.0, shas), _nat([1900.0], [2100.0])),
    }
    for (c, r), (ab, nat) in runs.items():
        (tmp_path / f"ab_{c}_{r}.json").write_text(json.dumps(ab))
        if nat is not None:
            (tmp_path / f"nat_{c}_{r}.json").write_text(json.dumps(nat))
    return tmp_path


def _run(out: Path, *extra: str) -> dict:
    args = [str(out), "--configs", "stock,on", "--rounds", "1,2", "--baseline", "stock",
            "--json", str(out / "t.json"), "--markdown", str(out / "t.md"), *extra]
    assert ab_table.main(args) == 0
    return json.loads((out / "t.json").read_text())


def test_metrics_by_run_means_and_ratios(out):
    s = _run(out)["summary"]
    assert s["prefill 8k"]["by_run"] == {"stock r1": 1050.0, "stock r2": 1250.0, "on r1": 2100.0, "on r2": 2500.0}
    assert s["prefill 8k first"]["by_run"]["on r2"] == 2400.0
    assert s["prefill 8k"]["mean"] == {"stock": 1150.0, "on": 2300.0}
    assert s["prefill 8k"]["ratio"] == {"on": pytest.approx(2.0)}
    assert set(s["prefill 8k"]["ratio"]) == {"on"}                  # no ratio for the baseline itself


def test_follow_ups_average_repeated_targets(out):
    s = _run(out)["summary"]
    assert s["follow-up 16k s"]["by_run"]["stock r1"] == pytest.approx(0.25)
    assert s["follow-up 16k filled"]["by_run"]["stock r1"] == pytest.approx(80.0)        # (60 + 100) / 2
    assert s["follow-up 16k filled"]["by_run"]["on r1"] == pytest.approx(134.0)          # (116 + 152) / 2
    assert s["follow-up 16k resumed at"]["by_run"]["on r2"] == pytest.approx(16416.0)
    assert s["follow-up 60k s"]["mean"]["on"] == pytest.approx(0.15)
    assert s["follow-up 16k s"]["ratio"]["on"] == pytest.approx(0.45 / 0.25)


def test_decode_metrics_and_acceptance_ignore_missing_values(out):
    s = _run(out)["summary"]
    assert s["decode 1 tok/s"]["by_run"]["stock r2"] == 42.0
    assert s["decode 1 MTP acceptance"]["by_run"]["on r1"] == pytest.approx(0.64)
    assert s["decode 3 wall tok/s"]["mean"]["stock"] == pytest.approx(77.5)
    assert s["decode 3 server sum"]["by_run"]["stock r1"] == pytest.approx(61.0)
    assert s["decode 3 MTP acceptance"]["by_run"]["stock r1"] == pytest.approx(0.5)      # the None stream left out
    assert s["decode 3 MTP acceptance"]["by_run"]["stock r2"] == pytest.approx(0.8)


def test_natural_prefill_and_a_missing_file(out):
    s = _run(out)["summary"]
    assert s["natural 32k cold"]["by_run"]["stock r1"] == 900.0
    assert s["natural 32k warm"]["by_run"]["stock r1"] == pytest.approx(1100.0)
    assert s["natural 32k warm"]["by_run"]["stock r2"] is None                           # no nat_stock_2.json
    assert s["natural 32k warm"]["mean"]["stock"] == pytest.approx(1100.0)              # over the runs that have it
    assert s["natural 32k cold"]["ratio"]["on"] == pytest.approx(1850.0 / 900.0)


def test_same_reply_tokens(out):
    same = _run(out)["same_tokens"]
    assert same["on r1 vs stock r1"] == "7/8"           # 2 prefill, 3 first turns, 1 + 2 decode replies
    assert same["on r2 vs stock r2"] == "8/8"
    assert same["stock r1 vs r2"] == "8/8"
    assert same["on r1 vs r2"] == "7/8"


def test_markdown_table_and_stdout(out, capsys):
    _run(out)
    md = (out / "t.md").read_text()
    printed = capsys.readouterr().out
    assert md.strip() in printed
    lines = md.strip().splitlines()
    assert lines[0] == "| metric | stock r1 | stock r2 | on r1 | on r2 | on/stock |"
    assert set(lines[1].replace("|", "").strip()) <= {"-", " ", ":"}
    assert "| prefill 8k | 1050 | 1250 | 2100 | 2500 | 2.000 |" in lines
    assert "| follow-up 16k s | 0.250 | 0.250 | 0.450 | 0.450 | 1.800 |" in lines
    assert "| natural 32k warm | 1100 | - | 2000 | 2100 | 1.864 |" in lines
    assert any(line.startswith("same reply tokens: ") and "on r1 vs stock r1 7/8" in line for line in lines)


def test_ledger_metrics_of_one_run(out):
    _run(out, "--ledger", str(out / "ledger.json"), "--ledger-run", "on:1")
    led = json.loads((out / "ledger.json").read_text())
    assert led == {"prefill_8k": 2100.0, "decode_1": 41.0, "decode_3": 76.0, "followup_s": 0.14}
