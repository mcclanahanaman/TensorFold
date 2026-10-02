"""tools/thor/qsa_digest.py: the QSA kernels' time inside each prompt range of an Nsight Systems SQLite export."""

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import qsa_digest  # noqa: E402

MS = 1_000_000          # ns


def _db(path: Path) -> Path:
    c = sqlite3.connect(path)
    c.execute("create table StringIds (id integer primary key, value text)")
    c.execute("create table NVTX_EVENTS (start integer, end integer, text text, textId integer)")
    c.execute("create table CUPTI_ACTIVITY_KIND_KERNEL (shortName integer, start integer, end integer, "
              "gridX integer, gridY integer, gridZ integer)")
    names = ["_chunks", "_merge", "_select", "front_kernel", "_scores", "_row_attn"]
    c.executemany("insert into StringIds values (?, ?)", list(enumerate(names, 1)) + [(50, "prompt 4096")])
    c.executemany("insert into NVTX_EVENTS values (?, ?, ?, ?)", [
        (0, 100 * MS, "prompt 2048", None),          # text inline
        (200 * MS, 400 * MS, None, 50),              # text through StringIds
        (10 * MS, 20 * MS, "pass 0", None),          # not a prompt range
    ])
    k = [  # (name id, start ms, end ms, gridZ)
        (1, 1, 5, 5), (1, 6, 8, 5), (1, 9, 10, 2), (2, 11, 12, 1), (4, 20, 60, 1), (3, 61, 63, 1),
        (1, 99, 101, 5),                             # ends past the range: not counted
        (1, 210, 230, 5), (5, 230, 231, 1), (6, 240, 250, 1), (4, 250, 300, 1),
    ]
    c.executemany("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, ?, 1, 1, ?)",
                  [(n, s * MS, e * MS, z) for n, s, e, z in k])
    c.commit()
    c.close()
    return path


def test_digest_splits_each_prompt_range_by_kernel(tmp_path):
    d = qsa_digest.digest(_db(tmp_path / "t.sqlite"))
    assert sorted(d) == ["prompt 2048", "prompt 4096"]
    a = d["prompt 2048"]
    assert a["kernel_ms"] == 4 + 2 + 1 + 1 + 40 + 2
    assert a["kernels"]["_chunks z5"] == {"n": 2, "ms": 6.0, "share": 6.0 / 50}
    assert a["kernels"]["_chunks z2"]["ms"] == 1.0 and a["kernels"]["_merge"]["n"] == 1
    assert a["kernels"]["_select"]["ms"] == 2.0 and "front_kernel" not in a["kernels"]
    assert a["qsa_ms"] == 10.0 and abs(a["qsa_share"] - 0.2) < 1e-12
    b = d["prompt 4096"]
    assert b["kernel_ms"] == 20 + 1 + 10 + 50
    assert set(b["kernels"]) == {"_chunks z5", "_scores", "_row_attn"}
    assert b["qsa_ms"] == 31.0


def test_names_limit_the_kernels_and_main_writes_json(tmp_path, capsys):
    db = _db(tmp_path / "t.sqlite")
    d = qsa_digest.digest(db, names=("_select",))
    assert set(d["prompt 2048"]["kernels"]) == {"_select"} and d["prompt 4096"]["kernels"] == {}
    out = tmp_path / "o.json"
    assert qsa_digest.main([str(db), "--json", str(out)]) == 0
    assert json.loads(out.read_text())["prompt 2048"]["qsa_ms"] == 10.0
    text = capsys.readouterr().out
    assert "prompt 2048" in text and "_chunks z5" in text
