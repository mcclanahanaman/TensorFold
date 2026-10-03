"""tools/thor/sparkdash_summary.py: repeated sparkDash runs (DECODE-0 driver JSONs) -> per config, type and stream
count means with 95% Student-t CIs, pooled MTP acceptance and tokens a round, parity verdicts against a baseline config
and a reference table, plus a digest of the server's per-request "done" lines."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import sparkdash_summary as ss  # noqa: E402


def srv(gen, req, drafted, accepted, rounds):
    return {"gen": gen, "requests": req, "drafted": drafted, "accepted": accepted, "rounds": rounds}


def wave(rep, kind, c, agg, per=None, ttft=70.0, ok=None, err=None, tokens=None, server=None):
    return {"rep": rep, "type": kind, "concurrency": c, "aggregateDecodeTps": agg,
            "meanDecodeTps": agg / c if per is None else per, "meanTtftMs": ttft,
            "streamsOk": c if ok is None else ok, "streamsFailed": 0 if ok is None else c - ok, "error": err,
            "totalCompletionTokens": 400 * c if tokens is None else tokens,
            "server": server or srv(None, c, None, None, None)}


def pre(rep, target, tps, tokens=8228, ttft=3400.0, err=None):
    return {"rep": rep, "targetTokens": target, "promptTokens": tokens, "prefillTps": tps, "ttftMs": ttft,
            "error": err}


A1 = {"label": "A_1", "waves": [
    wave(1, "prose", 1, 60.0, ttft=70.0, server=srv(400, 1, 300, 150, 249)),
    wave(1, "prose", 4, 110.0),
    wave(1, "code", 1, 95.0),
    wave(1, "code", 2, 100.0),
    wave(1, "structured", 2, 100.0),
    wave(2, "prose", 1, 62.0, ttft=66.0, server=srv(400, 1, 300, 180, 219)),
    wave(2, "prose", 4, 112.0),
    wave(2, "code", 2, 101.0),
    wave(2, "structured", 2, 110.0),
    wave(3, "prose", 1, 64.0, ttft=62.0, server=srv(400, 1, 250, 200, 199)),
], "prefill": [pre(1, 8192, 2400.0, ttft=3420.0), pre(2, 8192, 2420.0, ttft=3400.0)]}
A2 = {"label": "A_2", "waves": [
    wave(1, "prose", 1, 61.0, ttft=64.0, server=srv(400, 1, 200, 100, 299)),
    wave(2, "prose", 1, 63.0, ttft=68.0),
], "prefill": [pre(1, 8192, 2410.0, ttft=3410.0), pre(1, 32768, 2460.0, tokens=32805, ttft=13330.0)]}
B1 = {"label": "B_1", "waves": [
    wave(1, "prose", 1, 40.0), wave(2, "prose", 1, 41.0), wave(3, "prose", 1, 39.0), wave(4, "prose", 1, 40.0),
    wave(5, "prose", 1, 40.0),
    wave(1, "prose", 4, 109.0),
    wave(2, "prose", 4, 80.0, ok=3, err="1 of 4 streams failed"),
    wave(1, "code", 1, 50.0),
    wave(1, "code", 2, 120.0), wave(2, "code", 2, 122.0),
    wave(1, "structured", 2, 90.0), wave(2, "structured", 2, 92.0),
], "prefill": [pre(1, 8192, 3000.0), pre(2, 8192, 3060.0), pre(3, 8192, 0.0, err="No first token"),
               pre(1, 32768, 3150.0, tokens=32805, ttft=10414.0)]}
REF = {"decode": {"prose": {"1": 63.6, "4": 114.5}, "code": {"1": 96.9}}, "prefill": {"8192": 2421, "32768": 2462}}


def row(summary, config, kind, c):
    hits = [r for r in summary["decode"] if (r["config"], r["type"], r["conc"]) == (config, kind, c)]
    assert len(hits) == 1, (config, kind, c, hits)
    return hits[0]


def prow(summary, config, target):
    hits = [r for r in summary["prefill"] if (r["config"], r["target"]) == (config, target)]
    assert len(hits) == 1, (config, target, hits)
    return hits[0]


def test_t95_table():
    assert ss.t95(1) == 12.706
    assert ss.t95(2) == 4.303
    assert ss.t95(4) == 2.776
    assert ss.t95(9) == 2.262
    assert ss.t95(30) == 2.042
    assert ss.t95(35) == 2.021
    assert ss.t95(60) == 2.000
    assert ss.t95(100) == 1.980
    assert ss.t95(500) == 1.960


def test_ci95():
    c = ss.ci95([60.0, 62.0, 64.0, 61.0, 63.0])
    assert c["n"] == 5 and c["mean"] == pytest.approx(62.0)
    assert c["sd"] == pytest.approx(1.5811388, rel=1e-6)
    assert c["half"] == pytest.approx(2.776 * 1.5811388 / 5 ** 0.5, rel=1e-6)
    assert c["lo"] == pytest.approx(62.0 - c["half"]) and c["hi"] == pytest.approx(62.0 + c["half"])
    one = ss.ci95([109.0])
    assert one == {"n": 1, "mean": 109.0, "sd": None, "half": None, "lo": None, "hi": None}
    assert ss.ci95([]) == {"n": 0, "mean": None, "sd": None, "half": None, "lo": None, "hi": None}


def test_verdict():
    a = ss.ci95([100.0, 110.0])            # 105 [41.47, 168.53]
    b = ss.ci95([90.0, 92.0])              # 91 [78.29, 103.71]
    assert ss.verdict(b, a) == "overlap"
    assert ss.verdict(ss.ci95([109.0]), ss.ci95([110.0, 112.0])) == "within"
    assert ss.verdict(ss.ci95([50.0]), ss.ci95([95.0])) == "below"
    assert ss.verdict(ss.ci95([120.0, 122.0]), ss.ci95([100.0, 101.0])) == "above"
    assert ss.verdict(ss.ci95([]), ss.ci95([1.0])) == "n/a"
    assert ss.verdict(ss.ci95([103.0]), ss.ci95([100.0]), tol=0.03) == "within"
    assert ss.verdict(ss.ci95([103.5]), ss.ci95([100.0]), tol=0.03) == "above"


def test_summarize_decode_rows():
    s = ss.summarize([A1, A2, B1], baseline="A", reference=REF)
    keys = [(r["config"], r["type"], r["conc"]) for r in s["decode"]]
    assert keys == [("A", "prose", 1), ("A", "prose", 4), ("A", "code", 1), ("A", "code", 2),
                    ("A", "structured", 2),
                    ("B", "prose", 1), ("B", "prose", 4), ("B", "code", 1), ("B", "code", 2),
                    ("B", "structured", 2)]
    r = row(s, "A", "prose", 1)
    assert r["n"] == 5 and r["failed"] == 0
    assert r["agg"]["mean"] == pytest.approx(62.0) and r["agg"]["lo"] == pytest.approx(60.0371, abs=1e-4)
    assert r["per_req"]["mean"] == pytest.approx(62.0)
    assert r["ttft_ms"] == pytest.approx(66.0)
    assert r["tokens"] == pytest.approx(400.0)
    assert r["acceptance"] == pytest.approx(630 / 1050)
    assert r["tokens_per_round"] == pytest.approx(1596 / 966)
    assert r["by_round"] == {"1": pytest.approx(62.0), "2": pytest.approx(62.0)}
    assert "vs_baseline" not in r
    assert r["vs_reference_pct"] == pytest.approx(100 * 62.0 / 63.6)
    b = row(s, "B", "prose", 1)
    assert b["vs_baseline"]["pct"] == pytest.approx(100 * 40.0 / 62.0)
    assert b["vs_baseline"]["verdict"] == "below"
    assert b["acceptance"] is None and b["tokens_per_round"] is None
    b4 = row(s, "B", "prose", 4)
    assert b4["n"] == 1 and b4["failed"] == 1 and b4["agg"]["mean"] == pytest.approx(109.0)
    assert b4["per_req"]["mean"] == pytest.approx(109.0 / 4)
    assert b4["vs_baseline"]["verdict"] == "within"
    assert b4["vs_reference_pct"] == pytest.approx(100 * 109.0 / 114.5)
    assert row(s, "B", "structured", 2)["vs_baseline"]["verdict"] == "overlap"
    assert row(s, "B", "code", 2)["vs_baseline"]["verdict"] == "above"
    assert row(s, "B", "code", 1)["vs_baseline"]["verdict"] == "below"
    assert "vs_reference_pct" not in row(s, "B", "code", 2)


def test_summarize_prefill_rows():
    s = ss.summarize([A1, A2, B1], baseline="A", reference=REF)
    assert [(r["config"], r["target"]) for r in s["prefill"]] == [("A", 8192), ("A", 32768), ("B", 8192),
                                                                   ("B", 32768)]
    a = prow(s, "A", 8192)
    assert a["n"] == 3 and a["failed"] == 0
    assert a["tps"]["mean"] == pytest.approx(2410.0)
    assert a["tps"]["half"] == pytest.approx(4.303 * 10.0 / 3 ** 0.5, rel=1e-6)
    assert a["prompt_tokens"] == pytest.approx(8228.0) and a["ttft_ms"] == pytest.approx(3410.0)
    assert a["vs_reference_pct"] == pytest.approx(100 * 2410.0 / 2421)
    b = prow(s, "B", 8192)
    assert b["n"] == 2 and b["failed"] == 1
    assert b["vs_baseline"]["pct"] == pytest.approx(100 * 3030.0 / 2410.0)
    assert b["vs_baseline"]["verdict"] == "above"


def test_no_baseline_no_reference():
    s = ss.summarize([B1])
    r = row(s, "B", "prose", 1)
    assert "vs_baseline" not in r and "vs_reference_pct" not in r
    assert ss.summarize([{"label": "solo", "waves": [wave(1, "prose", 1, 10.0)], "prefill": []}])["decode"][0][
        "config"] == "solo"


def test_markdown_cells():
    md = ss.markdown(ss.summarize([A1, A2, B1], baseline="A", reference=REF))
    lines = md.splitlines()
    assert ("| config | type | streams | n | aggregate tok/s (95% CI) | per request tok/s (95% CI) | TTFT ms | "
            "acceptance | tokens/round | vs baseline | vs reference |") in lines
    assert ("| config | prompt | n | prefill tok/s (95% CI) | prompt tokens | TTFT ms | vs baseline | "
            "vs reference |") in lines
    a = [ln for ln in lines if ln.startswith("| A | prose | 1 |")]
    assert len(a) == 1
    assert "| 5 | 62.00 [60.04, 63.96] |" in a[0] and "| 66 | 0.600 | 1.65 | - | 97.5% |" in a[0]
    b = [ln for ln in lines if ln.startswith("| B | prose | 4 |")][0]
    assert "| 1 (1 failed) | 109.00 | 27.25 |" in b and "| 98.2% within | 95.2% |" in b
    p = [ln for ln in lines if ln.startswith("| A | 8192 |")][0]
    assert "| 3 | 2410.00 [2385.16, 2434.84] | 8228 | 3410 | - | 99.5% |" in p
    assert md.endswith("\n")


DONE = [
    "[tensorfold] done c1 prompt=37 cached=0 thinking=False effort=none tokens=400 sha=ab finish=length tok/s=60.1 "
    "ttft=0.07s prefill=0.05s rounds=180 accepted=219/320 ms/round=36.5 forward=30.0 draft=5.0 post=1.5 rows=3.2 "
    "checkpoints=off",
    "noise line",
    "[tensorfold] done c2 prompt=37 cached=0 thinking=False effort=none tokens=400 sha=cd finish=length tok/s=58.0 "
    "ttft=0.07s prefill=0.05s rounds=200 accepted=199/300 ms/round=38.5 forward=31.0 draft=6.0 post=1.5 rows=3.0 "
    "checkpoints=off",
    "[tensorfold] done c3 prompt=37 cached=0 thinking=False effort=none tokens=400 sha=ef finish=length tok/s=30.0 "
    "ttft=0.30s prefill=0.05s rounds=190 accepted=209/310 checkpoints=off",
    "[tensorfold] done c4 prompt=8228 cached=0 thinking=False effort=none tokens=8 sha=gh finish=length tok/s=9.0 "
    "ttft=3.40s prefill=3.39s rounds=4 accepted=3/6 ms/round=40.0 forward=33.0 draft=6.0 post=1.0 rows=2.0 "
    "checkpoints=off",
]


def test_digest_done():
    d = ss.digest_done(DONE, min_tokens=390)
    assert d["requests"] == 3 and d["profiled"] == 2
    assert d["ms_round"] == pytest.approx(37.5) and d["forward_ms"] == pytest.approx(30.5)
    assert d["draft_ms"] == pytest.approx(5.5) and d["post_ms"] == pytest.approx(1.5)
    assert d["rows"] == pytest.approx(3.1)
    assert d["tokens_per_round"] == pytest.approx(1197 / 570)
    assert d["acceptance"] == pytest.approx(627 / 930)
    assert d["server_tok_s"] == pytest.approx((60.1 + 58.0 + 30.0) / 3)
    e = ss.digest_done(["nothing here"], min_tokens=390)
    assert e["requests"] == 0 and e["profiled"] == 0 and e["ms_round"] is None and e["acceptance"] is None


def test_cli(tmp_path):
    (tmp_path / "d0_A_1.json").write_text(json.dumps(A1))
    (tmp_path / "d0_B_1.json").write_text(json.dumps(B1))
    (tmp_path / "other.json").write_text("{}")
    (tmp_path / "serve_A_1.log").write_text("\n".join(DONE) + "\n")
    (tmp_path / "ref.json").write_text(json.dumps(REF))
    out_j, out_m = tmp_path / "s.json", tmp_path / "s.md"
    rc = ss.main([str(tmp_path), "--json", str(out_j), "--markdown", str(out_m), "--baseline", "A",
                  "--reference", str(tmp_path / "ref.json")])
    assert rc == 0
    s = json.loads(out_j.read_text())
    assert {r["config"] for r in s["decode"]} == {"A", "B"}
    assert s["servers"]["A_1"]["requests"] == 3
    assert out_m.read_text().startswith("| config | type | streams |")
