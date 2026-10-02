"""Spec for tools/thor/followup_ttft.py (agentic follow-up turns on one server: TTFT, resumed and filled tokens).

    python -m pytest -q tools/thor/test_followup_ttft.py

Standard library and pytest only: a fake streaming server in this process; no GPU, no TensorFold import.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import followup_ttft  # noqa: E402
from prefill_ttft import text  # noqa: E402


def _tokens(messages: list) -> int:
    return sum(len(m["content"]) // 4 + 4 for m in messages)


class _Fake(BaseHTTPRequestHandler):
    """/v1/models and streamed chat completions; prompt tokens ~ characters / 4, cached = the messages before the last
    assistant message (where TensorFold resumes), a distinct reply per request; every request body is recorded."""

    seen: list = []

    def log_message(self, *args) -> None:
        pass

    def do_GET(self) -> None:
        body = json.dumps({"data": [{"id": "fake-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs = req["messages"]
        index = len(_Fake.seen)
        last = max([i for i, m in enumerate(msgs) if m["role"] == "assistant"], default=None)
        cached = 0 if last is None else _tokens(msgs[:last])
        reply = [f"r{index}"] + [f" w{i}" for i in range(req["max_tokens"] - 1)]
        sha = hashlib.sha256(json.dumps(msgs).encode()).hexdigest()[:12]
        _Fake.seen.append({"body": req, "reply": "".join(reply), "sha": sha, "prompt": _tokens(msgs),
                           "cached": cached})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def send(obj: dict) -> None:
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self.wfile.flush()

        send({"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]})
        for piece in reply:
            send({"choices": [{"delta": {"content": piece}, "finish_reason": None}]})
        send({"choices": [{"delta": {}, "finish_reason": "length"}],
              "usage": {"prompt_tokens": _tokens(msgs), "completion_tokens": len(reply),
                        "prompt_tokens_details": {"cached_tokens": cached}},
              "tensorfold": {"prefill_s": 0.01, "decode_s": 0.01, "rounds": 2, "drafted": 4, "accepted": 2,
                             "token_sha": sha}})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


@pytest.fixture
def server():
    _Fake.seen = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/"
    httpd.shutdown()
    httpd.server_close()


def _run(server: str, out: Path) -> dict:
    assert followup_ttft.main([server, str(out), "--lengths", "300,600", "--turns", "3", "--tool-tokens", "40",
                               "--reply", "5", "--seed", "7", "--label", "t"]) == 0
    return json.loads(out.read_text())


def test_conversations_turns_and_report_fields(server, tmp_path):
    r = _run(server, tmp_path / "f.json")
    assert r["model"] == "fake-model" and r["label"] == "t" and r["seed"] == 7
    assert r["lengths"] == [300, 600] and r["turns"] == 3 and r["tool_tokens"] == 40 and r["reply"] == 5
    assert [c["target"] for c in r["conversations"]] == [300, 600]
    assert [c["seed"] for c in r["conversations"]] == [8, 9]
    assert [[t["turn"] for t in c["turns"]] for c in r["conversations"]] == [[0, 1, 2, 3], [0, 1, 2, 3]]


def test_each_turn_extends_the_chat_with_the_reply_as_sent_and_a_new_tool_result(server, tmp_path):
    _run(server, tmp_path / "f.json")
    seen = _Fake.seen
    assert len(seen) == 8
    for conv, seed, n in ((0, 8, 300), (1, 9, 600)):
        reqs = seen[conv * 4:conv * 4 + 4]
        assert reqs[0]["body"]["messages"] == [{"role": "user", "content": text(n, seed)}]
        for t in range(1, 4):
            assert reqs[t]["body"]["messages"] == reqs[t - 1]["body"]["messages"] + [
                {"role": "assistant", "content": reqs[t - 1]["reply"]},
                {"role": "user", "content": "Tool result:\n" + text(40, seed * 1000 + t)}]
    for s in seen:
        body = s["body"]
        assert body["max_tokens"] == 5 and body["temperature"] == 0 and body["stream"] is True


def test_rows_carry_the_servers_figures(server, tmp_path):
    r = _run(server, tmp_path / "f.json")
    rows = [t for c in r["conversations"] for t in c["turns"]]
    for row, s in zip(rows, _Fake.seen):
        assert row["prompt_tokens"] == s["prompt"] and row["cached"] == s["cached"]
        assert row["filled"] == s["prompt"] - s["cached"]
        assert row["token_sha"] == s["sha"] and row["text"] == s["reply"] and row["completion_tokens"] == 5
        assert row["ttft_s"] >= 0 and row["total_s"] >= row["ttft_s"]
        assert not {"t0", "first", "end"} & set(row)
    for c in r["conversations"]:
        turns = c["turns"]
        assert turns[0]["cached"] == 0
        assert all(b["prompt_tokens"] > a["prompt_tokens"] and b["cached"] > a["cached"]
                   for a, b in zip(turns, turns[1:]))


def test_one_line_per_turn(server, tmp_path, capsys):
    r = _run(server, tmp_path / "f.json")
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("[followup t] ")]
    assert len(lines) == 8
    first = r["conversations"][1]["turns"][2]
    assert lines[6].startswith(f"[followup t] 600 turn 2: {first['prompt_tokens']} tokens, {first['cached']} cached "
                               f"({first['filled']} filled), TTFT ")
