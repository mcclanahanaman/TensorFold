"""Spec for tools/thor/testlog_summary.py (a summary of pytest -q logs: counts, failures by file, stack vs pristine).

    python -m pytest -q tools/thor/test_testlog_summary.py

Standard library and pytest only; no GPU, no TensorFold import.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import testlog_summary  # noqa: E402

LOG_A = """\
........................................................................ [ 50%]
..F..s.F                                                                 [100%]
=================================== FAILURES ===================================
________________________________ test_x[bf16] _________________________________
E   assert 1 == 2
----------------------------- Captured log call ------------------------------
ERROR    tensorfold.engine:engine.py:12 out of memory
tests/cuda/test_a.py:10: AssertionError
FAILED (this line is not in the summary section and does not count)
=========================== short test summary info ============================
SKIPPED [2] tests/cuda/test_b.py:33: needs two GPUs
FAILED tests/cuda/test_a.py::test_x[bf16] - assert 1 == 2
FAILED tests/cuda/test_c.py::test_y - RuntimeError: boom - with a dash
2 failed, 76 passed, 2 skipped, 5 warnings in 12.50s (0:00:12)
"""

LOG_B = """\
....E...                                                                 [100%]
==================================== ERRORS ====================================
ERROR tests/test_d.py::test_z - ModuleNotFoundError: No module named 'mlx'
ERROR tests/test_e.py
7 passed, 2 errors in 0.50s
"""

LOG_C = """\
...                                                                      [100%]
==================================== PASSES ====================================
=========================== short test summary info ============================
PASSED tests/test_f.py::test_1
PASSED tests/test_f.py::test_2
PASSED tests/test_g.py::test_3
============================== 3 passed in 0.01s ===============================
"""

LOG_D = """\
........................................................................ [ 10%]
.....
"""

LOG_E = "============================ no tests ran in 0.02s =============================\n"

PRISTINE = """\
=========================== short test summary info ============================
FAILED tests/cuda/test_a.py::test_x[bf16] - assert 1 == 2
FAILED tests/test_d.py::test_z - ModuleNotFoundError
1 passed, 2 failed in 3.00s
"""

ZERO = {"passed": 0, "failed": 0, "error": 0, "skipped": 0, "xfailed": 0, "xpassed": 0, "warnings": 0,
        "deselected": 0}


@pytest.fixture
def logs(tmp_path: Path) -> Path:
    for name, text in (("test_a.log", LOG_A), ("test_b.log", LOG_B), ("test_c.log", LOG_C), ("test_d.log", LOG_D),
                       ("test_e.log", LOG_E), ("pristine.log", PRISTINE)):
        (tmp_path / name).write_text(text)
    return tmp_path


def test_parse_log_counts_seconds_and_failures(logs):
    a = testlog_summary.parse_log((logs / "test_a.log").read_text())
    assert a["complete"] is True
    assert a["counts"] == {**ZERO, "passed": 76, "failed": 2, "skipped": 2, "warnings": 5}
    assert a["seconds"] == pytest.approx(12.5)
    assert a["failed"] == ["tests/cuda/test_a.py::test_x[bf16]", "tests/cuda/test_c.py::test_y"]
    b = testlog_summary.parse_log((logs / "test_b.log").read_text())
    assert b["counts"] == {**ZERO, "passed": 7, "error": 2}
    assert b["failed"] == ["tests/test_d.py::test_z", "tests/test_e.py"]       # no summary section: every line


def test_decorated_empty_and_truncated_logs(logs):
    c = testlog_summary.parse_log((logs / "test_c.log").read_text())
    assert c["counts"] == {**ZERO, "passed": 3} and c["seconds"] == pytest.approx(0.01) and c["failed"] == []
    d = testlog_summary.parse_log((logs / "test_d.log").read_text())
    assert d == {"complete": False, "counts": None, "seconds": None, "failed": [], "by_file": {}}
    e = testlog_summary.parse_log((logs / "test_e.log").read_text())
    assert e["complete"] is True and e["counts"] == ZERO and e["seconds"] == pytest.approx(0.02)


def test_by_file_from_the_summary_lines(logs):
    a = testlog_summary.parse_log((logs / "test_a.log").read_text())
    blank = {"passed": 0, "failed": 0, "error": 0, "skipped": 0, "xfailed": 0, "xpassed": 0}
    assert a["by_file"] == {"tests/cuda/test_a.py": {**blank, "failed": 1},
                            "tests/cuda/test_b.py": {**blank, "skipped": 2},
                            "tests/cuda/test_c.py": {**blank, "failed": 1}}
    c = testlog_summary.parse_log((logs / "test_c.log").read_text())
    assert c["by_file"] == {"tests/test_f.py": {**blank, "passed": 2}, "tests/test_g.py": {**blank, "passed": 1}}


def _run(logs: Path, *extra: str) -> dict:
    stack = [str(logs / n) for n in ("test_a.log", "test_b.log", "test_c.log", "test_d.log")]
    assert testlog_summary.main([*stack, "--json", str(logs / "s.json"), "--markdown", str(logs / "s.md"), *extra]) == 0
    return json.loads((logs / "s.json").read_text())


def test_totals_failures_and_files_over_the_stack_logs(logs):
    s = _run(logs)
    assert [x["path"] for x in s["logs"]] == [str(logs / n) for n in ("test_a.log", "test_b.log", "test_c.log",
                                                                      "test_d.log")]
    assert s["totals"] == {**ZERO, "passed": 86, "failed": 2, "error": 2, "skipped": 2, "warnings": 5}
    assert s["failed"] == ["tests/cuda/test_a.py::test_x[bf16]", "tests/cuda/test_c.py::test_y",
                           "tests/test_d.py::test_z", "tests/test_e.py"]
    assert list(s["by_file"]) == sorted(s["by_file"])
    assert s["by_file"]["tests/test_e.py"]["error"] == 1
    assert s["pristine"] is None and s["diff"] is None


def test_stack_against_pristine(logs):
    s = _run(logs, "--pristine", str(logs / "pristine.log"))
    assert s["pristine"]["failed"] == ["tests/cuda/test_a.py::test_x[bf16]", "tests/test_d.py::test_z"]
    assert s["diff"] == {"stack_only": ["tests/cuda/test_c.py::test_y", "tests/test_e.py"],
                         "both": ["tests/cuda/test_a.py::test_x[bf16]", "tests/test_d.py::test_z"],
                         "pristine_only": []}


def test_markdown(logs, capsys):
    _run(logs, "--pristine", str(logs / "pristine.log"))
    md = (logs / "s.md").read_text()
    assert md.strip() in capsys.readouterr().out
    lines = md.strip().splitlines()
    assert lines[0] == "| log | passed | failed | error | skipped | seconds |"
    assert set(lines[1].replace("|", "").strip()) <= {"-", " ", ":"}
    assert "| test_a.log | 76 | 2 | 0 | 2 | 12.5 |" in lines
    assert "| test_d.log | incomplete | - | - | - | - |" in lines
    assert "| total | 86 | 2 | 2 | 2 | 13.0 |" in lines
    assert "- tests/cuda/test_a.py: 1 failing (1 also on pristine)" in lines
    assert "- tests/cuda/test_c.py: 1 failing (0 also on pristine)" in lines
    assert "stack only: tests/cuda/test_c.py::test_y, tests/test_e.py" in lines
    _run(logs)
    plain = (logs / "s.md").read_text().strip().splitlines()
    assert "- tests/test_e.py: 1 failing" in plain and not any(x.startswith("stack only") for x in plain)
