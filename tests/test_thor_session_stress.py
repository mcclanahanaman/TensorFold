"""tools/thor/session_stress.py against a stub server (the 8-session dynamic stress).

The stub speaks the three routes the harness uses: GET /v1/capacity (state), POST /v1/capacity (a preview of one
request: its prompt tokens, the tokens it would resume from a kept prefix, and whether it is admissible now) and a
streamed POST /v1/chat/completions. Each session is named by "[s<i>] " at the start of its first user message, and the
stub scripts one behaviour per session so every classification the harness reports is exercised exactly once.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "thor" / "session_stress.py"


def _session(messages) -> int:
    first = next(m["content"] for m in messages if m["role"] == "user")
    return int(first.split("]", 1)[0].lstrip("[s"))


def _turn(messages) -> int:
    return sum(1 for m in messages if m["role"] == "user") - 1


def _tokens(messages) -> int:
    return sum(len(m["content"]) for m in messages if isinstance(m.get("content"), str)) // 4 + 10


class Stub:
    """Scripted behaviours, keyed by (session, turn):
    s1 t1: the first preview is not admissible (memory), the next one is.
    s2 t2: evicted before its preview (announced cold).
    s3 t1: the preview says warm, then the prefix is evicted before the request runs (raced, with an eviction).
    s4 t1: the first request is refused 503 with Retry-After: 1.
    s5 t3: the preview says the request would pass the window (length): the session ends there.
    s6: replies stream reasoning only (no content deltas).
    mode "bad": s0 t2 runs cold although announced warm and nothing was evicted (unexplained); s1 t1 is killed mid-stream.
    mode "overload": the extra sessions (index >= extras_from) never preview; each one's first request is refused 503
    with Retry-After: 1 and reason "slots", then served. Mode "overload-bad": the same refusal with neither the header
    nor a reason (a malformed refusal).
    """

    def __init__(self, mode: str = "good", extras_from: int = 1 << 30) -> None:
        self.mode = mode
        self.extras_from = extras_from
        self.lock = threading.Lock()
        self.kept: dict[int, int] = {}
        self.evicted = 0
        self.seen: set = set()
        self.live = 0
        self.previews: dict[int, int] = {}

    def preview(self, body: dict) -> dict:
        s, t, p = _session(body["messages"]), _turn(body["messages"]), _tokens(body["messages"])
        with self.lock:
            self.previews[s] = self.previews.get(s, 0) + 1
            if self.mode == "good" and (s, t) == (2, 2) and ("evict", s, t) not in self.seen:
                self.seen.add(("evict", s, t))
                self.kept.pop(2, None)
                self.evicted += 1
            if self.mode == "good" and (s, t) == (5, 3):
                return {"object": "capacity.preview", "prompt_tokens": p, "cached_tokens": 0, "admissible_now": False,
                        "reason": "length", "evicted_kept_sessions_total": self.evicted}
            if self.mode == "good" and (s, t) == (1, 1) and ("wait", s, t) not in self.seen:
                self.seen.add(("wait", s, t))
                return {"object": "capacity.preview", "prompt_tokens": p, "cached_tokens": self.kept.get(s, 0),
                        "admissible_now": False, "reason": "memory", "retry_after_s": 0.2,
                        "evicted_kept_sessions_total": self.evicted}
            return {"object": "capacity.preview", "prompt_tokens": p, "cached_tokens": self.kept.get(s, 0),
                    "admissible_now": True, "reason": None, "evicted_kept_sessions_total": self.evicted}

    def capacity(self) -> dict:
        with self.lock:
            return {"object": "capacity", "ok": True, "live_streams": self.live, "max_streams": 8,
                    "kept_sessions": [{"tokens": v * 1000} for v in self.kept.values()],
                    "evicted_kept_sessions_total": self.evicted, "overloaded": False}


def _handler(stub: Stub):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a) -> None:
            pass

        def _json(self, code: int, payload: dict, headers: dict | None = None) -> None:
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/v1/capacity":
                return self._json(200, stub.capacity())
            self._json(404, {"error": {"message": "no route"}})

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.rstrip("/") == "/v1/capacity":
                assert "stream" not in body, "a preview body carries no stream flag"
                return self._json(200, stub.preview(body))
            assert self.path.rstrip("/") == "/v1/chat/completions"
            assert body.get("stream") is True
            s, t, p = _session(body["messages"]), _turn(body["messages"]), _tokens(body["messages"])
            with stub.lock:
                refuse = None
                if stub.mode == "good" and (s, t) == (4, 1) and ("503", s, t) not in stub.seen:
                    stub.seen.add(("503", s, t))
                    refuse = ({"error": {"message": "busy", "reason": "memory"}}, {"Retry-After": "1"})
                elif stub.mode.startswith("overload") and s >= stub.extras_from and ("503", s) not in stub.seen:
                    stub.seen.add(("503", s))
                    refuse = (({"error": {"message": "busy", "reason": "slots"}}, {"Retry-After": "1"})
                              if stub.mode == "overload" else ({"error": {"message": "busy"}}, {}))
                if stub.mode == "good" and (s, t) == (3, 1):
                    stub.kept.pop(3, None)
                    stub.evicted += 1
                if stub.mode == "bad" and (s, t) == (0, 2):
                    stub.kept.pop(0, None)                  # cold, announced warm, no eviction counted
                cached = stub.kept.get(s, 0)
                stub.live += 1
            if refuse is not None:
                with stub.lock:
                    stub.live -= 1
                return self._json(503, *refuse)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            key = "reasoning_content" if s == 6 else "content"

            def send(obj) -> None:
                self.wfile.write(b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n\n")
                self.wfile.flush()

            for piece in ("a", "b", "c"):
                time.sleep(0.01)
                send({"choices": [{"index": 0, "delta": {key: piece}}]})
                if stub.mode == "bad" and (s, t) == (1, 1):
                    send({"error": {"message": "stopped for memory", "type": "server_error"}})
                    with stub.lock:
                        stub.live -= 1
                    return
            send({"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                  "usage": {"prompt_tokens": p, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": cached}}})
            send(b"[DONE]")
            with stub.lock:
                stub.kept[s] = p
                stub.live -= 1

    return H


def _run(tmp_path, mode: str, sessions: int, turns: int, *extra: str,
         stub: Stub | None = None) -> tuple[int, dict, str]:
    stub = stub or Stub(mode)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _handler(stub))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    out = tmp_path / f"{mode}.json"
    try:
        proc = subprocess.run([sys.executable, str(TOOL), f"http://127.0.0.1:{srv.server_address[1]}", str(out),
                               "--sessions", str(sessions), "--turns", str(turns), "--plan", "tiny", "--seed", "7",
                               "--max-wait", "20", "--sample-s", "0.2", "--stagger-s", "0", "--turn-timeout", "30",
                               *extra],
                              capture_output=True, text=True, timeout=120)
    finally:
        srv.shutdown()
    return proc.returncode, json.loads(out.read_text()), proc.stdout


def test_every_classification_once(tmp_path):
    rc, doc, stdout = _run(tmp_path, "good", 7, 4)
    s = doc["summary"]
    assert (s["sessions"], s["turns_sent"], s["followups"]) == (7, 27, 20)
    assert (s["warm"], s["announced_cold"], s["raced"], s["unexplained"]) == (18, 1, 1, 0)
    assert (s["errors"], s["killed"], s["timeouts"], s["length_ended"], s["retries"]) == (0, 0, 0, 1, 1)
    assert (s["refused"], s["refused_by"], s["refusals_malformed"], s["extra_sessions"]) == (1, {"memory": 1}, 0, 0)
    assert s["waited_s"] > 0 and s["held_long_max"] >= 1
    assert s["pass"] is True and rc == 0
    assert "session_stress: PASS" in stdout
    rows = {(r["session"], r["turn"]): r for r in doc["rows"]}
    assert rows[(2, 2)]["class"] == "announced_cold" and rows[(3, 1)]["class"] == "raced"
    assert rows[(5, 3)].get("ended") == "length" and (5, 4) not in rows
    assert rows[(4, 1)]["retries"] == 1 and rows[(1, 1)]["waited_s"] > 0
    assert [(x["code"], x["reason"], x["retry_after"]) for x in rows[(4, 1)]["refusals"]] == [(503, "memory", "1")]
    assert all(r["refusals"] == [] and r["extra"] is False
               for k, r in rows.items() if k != (4, 1) and not r.get("ended"))
    assert all(rows[(6, t)]["ttft_s"] is not None for t in range(4))
    assert all(r["class"] == "warm" for k, r in rows.items() if k[1] > 0 and k not in ((2, 2), (3, 1), (5, 3)))
    for r in doc["rows"]:
        if r.get("ended"):
            continue
        assert {"prompt_tokens", "cached_tokens", "prev_prompt_tokens", "preview_prompt_tokens",
                "preview_cached_tokens", "ttft_s", "total_s", "completion_tokens", "retries", "waited_s"} <= r.keys()
    assert doc["timeline"] and {"t", "live_streams", "kept_tokens", "evicted_kept_sessions_total"} <= doc["timeline"][0].keys()


def test_unexplained_cold_and_a_kill_fail(tmp_path):
    rc, doc, stdout = _run(tmp_path, "bad", 2, 3)
    s = doc["summary"]
    assert s["unexplained"] == 1 and s["killed"] == 1
    assert s["pass"] is False and rc == 1
    assert "session_stress: FAIL" in stdout


def test_extra_sessions_skip_the_preview_and_back_off_on_a_503_with_retry_after(tmp_path):
    stub = Stub("overload", extras_from=2)
    rc, doc, stdout = _run(tmp_path, "overload", 2, 2, "--extra", "2", "--extra-turns", "2", "--require-refusal",
                           stub=stub)
    s = doc["summary"]
    assert (s["sessions"], s["extra_sessions"], s["turns_sent"], s["extra_turns_done"]) == (4, 2, 8, 4)
    assert (s["refused"], s["refused_by"], s["refusals_malformed"]) == (2, {"slots": 2}, 0)
    assert (s["errors"], s["killed"], s["timeouts"], s["retries"]) == (0, 0, 0, 2)
    assert s["pass"] is True and rc == 0 and "session_stress: PASS" in stdout
    assert stub.previews.get(2, 0) == 0 and stub.previews.get(3, 0) == 0      # an extra session never previews
    assert stub.previews.get(0, 0) >= 2 and stub.previews.get(1, 0) >= 2
    rows = {(r["session"], r["turn"]): r for r in doc["rows"]}
    for i in (2, 3):
        first = rows[(i, 0)]
        assert first["extra"] is True and first["retries"] == 1 and first["waited_s"] >= 1
        assert [(x["code"], x["reason"], x["retry_after"]) for x in first["refusals"]] == [(503, "slots", "1")]
        assert first["refusals"][0]["at_s"] >= 0
        assert rows[(i, 1)]["refusals"] == [] and rows[(i, 1)]["class"] == "warm"
    assert rows[(0, 0)]["extra"] is False and rows[(0, 0)]["refusals"] == []


def test_a_refusal_without_retry_after_or_reason_is_malformed_and_fails(tmp_path):
    stub = Stub("overload-bad", extras_from=1)
    rc, doc, stdout = _run(tmp_path, "overload-bad", 1, 1, "--extra", "1", "--extra-turns", "1", stub=stub)
    s = doc["summary"]
    assert (s["refused"], s["refusals_malformed"], s["errors"]) == (1, 1, 0)
    rows = {(r["session"], r["turn"]): r for r in doc["rows"]}
    assert [(x["code"], x["reason"], x["retry_after"]) for x in rows[(1, 0)]["refusals"]] == [(503, None, None)]
    assert s["pass"] is False and rc == 1 and "session_stress: FAIL" in stdout


def test_require_refusal_fails_a_run_that_saw_none(tmp_path):
    rc, doc, stdout = _run(tmp_path, "good", 2, 2, "--require-refusal")
    s = doc["summary"]
    assert s["refused"] == 0 and s["errors"] == 0 and s["killed"] == 0
    assert s["pass"] is False and rc == 1


def test_an_extra_session_s_cold_turn_is_unpreviewed_not_announced():
    import importlib.util
    spec = importlib.util.spec_from_file_location("session_stress", TOOL)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    base = {'error': None, 'killed_mid_stream': False, 'timeout': False, 'retries': 0, 'waited_s': 0.0,
            'prompt_tokens': 100, 'cached_tokens': 0, 'refusals': []}
    rows = [{**base, 'session': 0, 'turn': 1, 'class': 'announced_cold', 'extra': False},
            {**base, 'session': 9, 'turn': 1, 'class': 'unpreviewed_cold', 'extra': True}]
    s = tool.summarize(rows, [])
    assert (s['announced_cold'], s['unpreviewed_cold'], s['unexplained'], s['followups']) == (1, 1, 0, 2)
    assert s['pass'] is True
