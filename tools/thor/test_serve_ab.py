"""Spec for serve_ab.py (DEC-2): an A/B of sparkDash serve drives (d0_drive.py JSON, replies kept as replySha)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def wave(kind, conc, rep, tps, shas, error=None, ok=None):
    return {"type": kind, "concurrency": conc, "rep": rep, "aggregateDecodeTps": tps, "error": error,
            "streamsOk": conc if ok is None else ok,
            "streams": [{"index": i, "replySha": s, "replyChars": 10} for i, s in enumerate(shas)]}


def drive(tmp_path, name, waves):
    p = tmp_path / name
    p.write_text(json.dumps({"label": name, "waves": waves}))
    return str(p)


@pytest.fixture
def files(tmp_path):
    o1 = drive(tmp_path, "o1.json", [wave("prose", 1, 1, 50.0, ["a"]), wave("prose", 1, 2, 52.0, ["a"]),
                                     wave("prose", 2, 1, 80.0, ["b", "c"]), wave("code", 2, 1, 120.0, ["d", "e"])])
    n1 = drive(tmp_path, "n1.json", [wave("prose", 1, 1, 55.0, ["a"]), wave("prose", 1, 2, 53.0, ["a"]),
                                     wave("prose", 2, 1, 84.0, ["b", "c"]), wave("code", 2, 1, 126.0, ["d", "X"])])
    n2 = drive(tmp_path, "n2.json", [wave("prose", 1, 1, 56.0, ["a"]), wave("prose", 1, 2, 54.0, ["a"]),
                                     wave("prose", 2, 1, 86.0, ["b", "c"]),
                                     wave("code", 2, 1, 999.0, ["d", "e"], error="1 of 2 streams failed", ok=1)])
    o2 = drive(tmp_path, "o2.json", [wave("prose", 1, 1, 51.0, ["a"]), wave("prose", 1, 2, 51.0, ["Q"]),
                                     wave("prose", 2, 1, 82.0, ["b", "c"]), wave("code", 2, 1, 118.0, ["d", "e"])])
    return {"O": [o1, o2], "N": [n1, n2]}


def test_cells_ratio_pairs(files):
    import serve_ab

    r = serve_ab.summarize(files, base="O")
    c = r["cells"]["prose"]["1"]
    assert c["O"]["tok_s"] == pytest.approx((50 + 52 + 51 + 51) / 4)
    assert c["O"]["session_means"] == pytest.approx([51.0, 51.0])
    assert c["N"]["tok_s"] == pytest.approx((55 + 53 + 56 + 54) / 4)
    assert c["N"]["n"] == 4
    assert c["N"]["ratio"] == pytest.approx(54.5 / 51.0)
    assert c["N"]["pairs"] == pytest.approx([54.0 / 51.0, 55.0 / 51.0])
    assert "ratio" not in c["O"]


def test_failed_waves_dropped(files):
    import serve_ab

    r = serve_ab.summarize(files, base="O")
    c = r["cells"]["code"]["2"]
    assert c["N"]["n"] == 1 and c["N"]["dropped"] == 1
    assert c["N"]["tok_s"] == pytest.approx(126.0)
    assert c["N"]["session_means"] == [pytest.approx(126.0), None]
    assert c["N"]["pairs"] == [pytest.approx(126.0 / 120.0), None]


def test_identity(files):
    import serve_ab

    ident = serve_ab.summarize(files, base="O")["identity"]
    # replies keyed (type, concurrency, rep, stream index); a failed wave's replies are not compared.
    # cross: every (other-arm session, base session) pair; within: every pair of sessions of one arm.
    assert ident["cross"] == {"compared": 6 + 6 + 4 + 4, "mismatched": 1 + 2 + 0 + 1}
    assert ident["within"] == {"compared": 6 + 4, "mismatched": 1}
    assert ident["identical"] is False
    assert ["code", 2, 1, 1] in ident["examples"] and ["prose", 1, 2, 0] in ident["examples"]


def test_cli(files, tmp_path):
    out = tmp_path / "ab.json"
    args = [sys.executable, str(HERE / "serve_ab.py"), str(out), "--arm", "O=" + ",".join(files["O"]),
            "--arm", "N=" + ",".join(files["N"]), "--base", "O"]
    p = subprocess.run(args, capture_output=True, text=True, check=True)
    lines = [l for l in p.stdout.splitlines() if l.startswith("[ab]")]
    assert any("prose c1" in l and "N/O 1.069" in l for l in lines)
    assert any(l.startswith("[ab] identity") and "cross 4/20" in l and "within 1/10" in l for l in lines)
    got = json.loads(out.read_text())
    assert got["base"] == "O" and set(got["cells"]) == {"prose", "code"}
