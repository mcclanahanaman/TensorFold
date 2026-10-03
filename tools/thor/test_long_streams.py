"""Spec for tools/thor/long_streams.py: concurrent long-prompt streams (the THOR-DEC-1 full-window stress). CPU only."""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prefill_ttft import text  # noqa: E402


def _tool():
    spec = importlib.util.spec_from_file_location("long_streams", HERE / "long_streams.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake(n, fail_seed=None):
    calls, lock, barrier = [], threading.Lock(), threading.Barrier(n, timeout=10)

    def chat(base, model, messages, max_tokens, *, ignore_eos=False, timeout=3600.0):
        with lock:
            calls.append({"base": base, "model": model, "messages": messages, "max_tokens": max_tokens,
                          "ignore_eos": ignore_eos, "timeout": timeout})
        barrier.wait()                              # every stream in flight at once, or this raises
        content = messages[0]["content"]
        if fail_seed is not None and content.startswith(f"Run {fail_seed}."):
            raise RuntimeError("the server stream ended with an error: out of memory")
        return {"t0": 1.0, "first": 2.0, "end": 3.0, "ttft_s": 1.0, "total_s": 2.0, "prompt_tokens": len(content),
                "cached": 0, "completion_tokens": max_tokens, "server_decode_tok_s": 30.0, "token_sha": "ab",
                "text": "x"}

    return calls, chat


def _patch(monkeypatch, mod, chat):
    monkeypatch.setattr(mod, "chat", chat)
    monkeypatch.setattr(mod, "model_id", lambda base: "m")


def test_run_streams_at_once(monkeypatch):
    mod = _tool()
    calls, chat = _fake(3)
    _patch(monkeypatch, mod, chat)
    rep = mod.run("http://h:1/", [300, 200, 100], seed=7, max_tokens=16, timeout=99.0, label="L")
    assert len(calls) == 3
    assert {c["messages"][0]["content"] for c in calls} == {text(300, 7), text(200, 8), text(100, 9)}
    for c in calls:
        assert c["base"] == "http://h:1" and c["model"] == "m" and c["max_tokens"] == 16
        assert c["ignore_eos"] is True and c["timeout"] == 99.0
        assert c["messages"][0]["role"] == "user" and len(c["messages"]) == 1
    assert (rep["label"], rep["url"], rep["model"]) == ("L", "http://h:1", "m")
    assert (rep["lengths"], rep["seed"], rep["max_tokens"], rep["errors"]) == ([300, 200, 100], 7, 16, 0)
    assert [s["i"] for s in rep["streams"]] == [0, 1, 2]
    for s, n, seed in zip(rep["streams"], [300, 200, 100], [7, 8, 9]):
        assert s["target"] == n and s["prompt_tokens"] == len(text(n, seed)) and s["completion_tokens"] == 16
        assert not {"t0", "first", "end"} & set(s) and "error" not in s
    assert isinstance(rep["wall_s"], float) and rep["wall_s"] >= 0.0


def test_run_keeps_a_failed_stream(monkeypatch):
    mod = _tool()
    calls, chat = _fake(3, fail_seed=8)
    _patch(monkeypatch, mod, chat)
    rep = mod.run("http://h:1", [30, 20, 10], seed=7, max_tokens=4, timeout=5.0)
    assert rep["errors"] == 1 and rep["label"] == ""
    bad = rep["streams"][1]
    assert bad["i"] == 1 and bad["target"] == 20 and "out of memory" in bad["error"] and "prompt_tokens" not in bad
    assert "error" not in rep["streams"][0] and "error" not in rep["streams"][2]


def test_defaults():
    a = _tool().parse_args(["http://h:1", "o.json"])
    assert (a.url, str(a.out), a.label) == ("http://h:1", "o.json", "")
    assert a.lengths == [240000, 220000, 200000, 180000]
    assert (a.seed, a.max_tokens, a.timeout) == (20261003, 256, 7200.0)


def test_main_writes_and_exit_code(monkeypatch, tmp_path):
    mod = _tool()
    out = tmp_path / "o.json"
    _patch(monkeypatch, mod, _fake(2)[1])
    rc = mod.main(["http://h:1", str(out), "--lengths", "50,40", "--seed", "3", "--max-tokens", "8", "--label", "x"])
    data = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0 and data["lengths"] == [50, 40] and data["label"] == "x" and data["max_tokens"] == 8
    _patch(monkeypatch, mod, _fake(2, fail_seed=3)[1])
    assert mod.main(["http://h:1", str(out), "--lengths", "50,40", "--seed", "3"]) == 1
    assert json.loads(out.read_text(encoding="utf-8"))["errors"] == 1
