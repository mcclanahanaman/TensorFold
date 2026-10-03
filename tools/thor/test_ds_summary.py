"""tools/thor/ds_summary.py: DEC-STRUCT tables from ds_drive JSON runs (every stream's reply text and per-request MTP
stats): wave streams vs their 1-stream reference, per-cell means, reference stability and cross-run reference identity."""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ds_summary as ds  # noqa: E402

P1 = "Count. (stream 1/2)"
P2 = "Count. (stream 2/2)"


def rec(phase, prompt, text, tok, rounds, drafted, accepted, *, conc=1, rep=0, stream=0, same=None, diff_at=None,
        kind="structured", cached=0):
    tpr = round((tok - 1) / rounds, 3)
    acc = round(accepted / drafted, 3) if drafted else None
    return {"phase": phase, "type": kind, "conc": conc, "rep": rep, "stream": stream, "prompt": prompt, "text": text,
            "reasoning": "", "error": None, "same_as_ref1": same, "diff_at": diff_at,
            "d": {"comp": tok, "rounds": rounds, "drafted": drafted, "accepted": accepted, "cached": cached,
                  "tpr": tpr, "acc": acc, "dpr": round(drafted / rounds, 3)}}


def run(label, text2_wave="1 2 3 x", text2_ref="1 2 3 x"):
    return {"label": label, "records": [
        rec("ref1", P1, "1 2 3 4", 400, 57, 342, 342),
        rec("ref1", P2, text2_ref, 400, 235, 1410, 164, stream=1),
        rec("wave", P1, "1 2 3 4", 400, 57, 342, 342, conc=2, rep=1, stream=0, same=True, diff_at=-1),
        rec("wave", P2, text2_wave, 400, 133, 798, 266, conc=2, rep=1, stream=1,
            same=text2_wave == text2_ref, diff_at=-1 if text2_wave == text2_ref else 6),
        rec("ref2", P1, "1 2 3 4", 400, 57, 342, 342, same=True, cached=8),
        rec("ref2", P2, text2_ref, 400, 235, 1410, 164, stream=1, same=True, cached=8),
    ]}


def test_wave_rows_pair_each_stream_with_its_reference():
    rows = ds.wave_rows(run("B"))
    assert [(r["conc"], r["stream"]) for r in rows] == [(2, 0), (2, 1)]
    r = rows[1]
    assert r["label"] == "B" and r["type"] == "structured" and r["rep"] == 1
    assert r["tok"] == 400 and r["rounds"] == 133 and r["tpr"] == 3.0 and r["acc"] == round(266 / 798, 3)
    assert r["ref_tpr"] == round(399 / 235, 3) and r["ref_acc"] == round(164 / 1410, 3)
    assert r["same"] is True and r["diff_at"] == -1


def test_cell_summary_counts_identical_streams_and_means():
    cells = ds.cell_summary(run("B", text2_wave="1 2 3 y"))
    assert len(cells) == 1
    c = cells[0]
    assert (c["type"], c["conc"], c["streams"], c["identical"]) == ("structured", 2, 2, 1)
    assert c["tpr_mean"] == round((7.0 + 3.0) / 2, 3) and c["tpr_min"] == 3.0 and c["tpr_max"] == 7.0
    assert c["ref_tpr_mean"] == round((7.0 + round(399 / 235, 3)) / 2, 3)


def test_reference_stability_reports_ref2_against_ref1():
    st = ds.ref_stability(run("B"))
    assert [(s["prompt"], s["same"], s["cached"]) for s in st] == [(P1, True, 8), (P2, True, 8)]


def test_cross_compares_ref1_text_of_the_same_prompt_across_runs():
    rows = ds.cross([run("B"), run("S", text2_wave="1 2 3 z", text2_ref="1 2 3 z")])
    by = {r["prompt"]: r for r in rows}
    assert by[P1]["identical"] is True and by[P1]["diff_at"] == -1
    assert by[P2]["identical"] is False and by[P2]["diff_at"] == 6
    assert by[P2]["labels"] == ["B", "S"] and by[P2]["tpr"] == [round(399 / 235, 3)] * 2


def test_snippet_escapes_newlines_and_keeps_head_and_tail():
    assert ds.snippet("a\nb", 10) == "a\\nb"
    s = ds.snippet("0123456789" * 3, 5)
    assert s == "01234 ... 56789"


def test_markdown_and_cli(tmp_path):
    md = ds.markdown([run("B"), run("S")])
    for want in ("## B: wave streams", "## S: per cell", "## references: ref2 vs ref1", "## ref1 across runs",
                 "| structured | 2 | 2 | 2 |", "## structured ref1 text"):
        assert want in md, want
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(run("B")))
    b.write_text(json.dumps(run("S")))
    out_md, out_js = tmp_path / "o.md", tmp_path / "o.json"
    subprocess.run([sys.executable, str(Path(__file__).parent / "ds_summary.py"), str(a), str(b), "--md", str(out_md),
                    "--json", str(out_js)], check=True, capture_output=True)
    assert "## B: wave streams" in out_md.read_text()
    js = json.loads(out_js.read_text())
    assert set(js) == {"waves", "cells", "refs", "cross"} and len(js["waves"]) == 4 and len(js["cross"]) == 2
