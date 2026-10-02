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


# --- card tk-ab-decode: decode at several stream counts (serve_ab.py --streams 2,3,4) and the follow-ups' first turns


def _ab_streams(by: dict, one_sha: str, first_ttft: list) -> dict:
    """A serve_ab.py report with ``decode.by_streams`` ({count: [(tok_s, acceptance, sha), ...]}, wall = 10 x count)
    and follow-ups whose first turns carry ``ttft_s`` ([(target, ttft), ...])."""

    by_streams = {str(n): {"wall_tok_s": 10.0 * n, "server_sum_tok_s": sum(v for v, _, _ in rows),
                           "streams": [{"server_decode_tok_s": v, "acceptance": a, "token_sha": h}
                                       for v, a, h in rows]} for n, rows in by.items()}
    many = by_streams.get("3") or by_streams[str(max(by))]
    return {"prefill": [{"target": 8192, "round": 0, "tok_s": 1000.0, "token_sha": "p"}],
            "followups": [{"target": t, "first": {"ttft_s": s, "token_sha": f"f{t}"},
                           "next": {"ttft_s": 0.2, "prompt_tokens": t + 100, "cached": t}} for t, s in first_ttft],
            "decode": {"one": {"server_decode_tok_s": 40.0, "acceptance": 0.6, "token_sha": one_sha},
                       "many": many, "by_streams": by_streams}}


@pytest.fixture
def streams_out(tmp_path: Path) -> Path:
    by = {4: [(20.0, 0.5, "d"), (21.0, 0.7, "e"), (22.0, None, "f"), (23.0, 0.9, "g")],
          2: [(30.0, 0.4, "a"), (31.0, 0.6, "b")],
          3: [(25.0, None, "a"), (26.0, None, "b"), (27.0, None, "c")]}
    first = [(16384, 8.0), (61440, 37.0), (16384, 10.0)]
    (tmp_path / "ab_stock_1.json").write_text(json.dumps(_ab_streams(by, "o", first)))
    by_on = {n: [(v + 1.0, a, h) for v, a, h in rows] for n, rows in by.items()}
    by_on[4][3] = (24.0, 0.9, "x")                                  # one of the 4-stream replies differs
    (tmp_path / "ab_on_1.json").write_text(json.dumps(_ab_streams(by_on, "o", [(16384, 4.0), (61440, 18.5),
                                                                                (16384, 5.0)])))
    return tmp_path


def _run_streams(out: Path, *extra: str) -> dict:
    args = [str(out), "--configs", "stock,on", "--rounds", "1", "--baseline", "stock",
            "--json", str(out / "t.json"), "--markdown", str(out / "t.md"), *extra]
    assert ab_table.main(args) == 0
    return json.loads((out / "t.json").read_text())


def test_decode_by_stream_count_replaces_the_single_many_entry(streams_out):
    s = _run_streams(streams_out)["summary"]
    keys = list(s)
    decode = [k for k in keys if k.startswith("decode ")]
    assert decode == ["decode 1 tok/s", "decode 1 MTP acceptance",
                      "decode 2 wall tok/s", "decode 2 server sum", "decode 2 MTP acceptance",
                      "decode 3 wall tok/s", "decode 3 server sum",
                      "decode 4 wall tok/s", "decode 4 server sum", "decode 4 MTP acceptance"]
    assert s["decode 2 wall tok/s"]["by_run"]["stock r1"] == pytest.approx(20.0)
    assert s["decode 4 server sum"]["by_run"]["stock r1"] == pytest.approx(86.0)
    assert s["decode 4 server sum"]["by_run"]["on r1"] == pytest.approx(90.0)
    assert s["decode 4 MTP acceptance"]["by_run"]["stock r1"] == pytest.approx(0.7)      # (0.5 + 0.7 + 0.9) / 3
    assert s["decode 2 MTP acceptance"]["by_run"]["stock r1"] == pytest.approx(0.5)
    assert s["decode 4 server sum"]["ratio"]["on"] == pytest.approx(90.0 / 86.0)


def test_first_turn_ttft_of_each_follow_up_target(streams_out):
    s = _run_streams(streams_out)["summary"]
    assert s["first turn 16k s"]["by_run"] == {"stock r1": pytest.approx(9.0), "on r1": pytest.approx(4.5)}
    assert s["first turn 60k s"]["by_run"]["on r1"] == pytest.approx(18.5)
    assert s["first turn 60k s"]["ratio"]["on"] == pytest.approx(18.5 / 37.0)
    keys = list(s)
    assert keys.index("first turn 16k s") < keys.index("follow-up 16k s") < keys.index("first turn 60k s")
    assert "first turn 16k s" not in _run(_legacy_out(streams_out))["summary"]   # first turns without ttft_s


def _legacy_out(streams_out: Path) -> Path:
    """The tk-ab-table fixture's reports (first turns without ttft_s, decode without by_streams) in a sub-folder."""

    sub = streams_out / "legacy"
    sub.mkdir()
    shas = [f"h{i}" for i in range(8)]
    for c in ("stock", "on"):
        for r in (1, 2):
            ab = _ab({8192: [1000.0, 1100.0]}, [(16384, 0.20, 16500, 16440)], (40.0, 0.6),
                     [(30.0, 0.5), (31.0, 0.5), (32.0, 0.5)], 75.0, shas[:7])
            (sub / f"ab_{c}_{r}.json").write_text(json.dumps(ab))
    return sub


def test_same_reply_tokens_follow_the_stream_counts_in_order(streams_out):
    same = _run_streams(streams_out)["same_tokens"]
    # prefill 1 + first turns 3 + decode.one 1 + streams 2 + 3 + 4 (counts in ascending order) = 14; one differs
    assert same == {"on r1 vs stock r1": "13/14"}


def test_ledger_takes_users_4_from_the_4_stream_wall_rate(streams_out):
    _run_streams(streams_out, "--ledger", str(streams_out / "ledger.json"), "--ledger-run", "on:1")
    led = json.loads((streams_out / "ledger.json").read_text())
    assert led == {"prefill_8k": 1000.0, "decode_1": 40.0, "decode_3": 30.0, "users_4": 40.0, "followup_s": 0.2}
