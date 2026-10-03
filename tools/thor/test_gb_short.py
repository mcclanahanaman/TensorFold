"""Spec for tools/thor/gb_short.py: the short greedy G-B probe (THOR-DEC-1). CPU only, a stub server, a stub sparkDash."""

from __future__ import annotations

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUB = '''
DEFAULT_MAX_TOKENS = 400
def pick_decode_bench_prompts(count, type_name):
    return [f"{type_name} p{i}" for i in range(count)]
def apply_thinking_flags(body, model_id, think=False):
    body = dict(body)
    body["chat_template_kwargs"] = {"enable_thinking": think}
    return body
'''


def _tool():
    spec = importlib.util.spec_from_file_location("gb_short", HERE / "gb_short.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Server:
    def __init__(self) -> None:
        self.bodies, self.live, self.most = [], 0, 0
        lock, outer = threading.Lock(), self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj):
                data = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send({"data": [{"id": "stub-model"}]})

            def do_POST(self):
                with lock:
                    outer.live += 1
                    outer.most = max(outer.most, outer.live)
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.bodies.append((self.path, body))
                text = "echo:" + body["messages"][0]["content"] + ":" + str(body["max_tokens"]) + " tail of the answer"
                self._send({"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "length"}],
                            "usage": {"prompt_tokens": 7, "completion_tokens": 5}})
                with lock:
                    outer.live -= 1

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


def _collect(tmp_path, mod, server):
    stub = tmp_path / "sd_stub.py"
    stub.write_text(STUB)
    out = tmp_path / "a.json"
    rc = mod.main(["collect", server.url, str(out), "--sparkdash", str(stub), "--types", "prose,code",
                   "--per-type", "2", "--max-tokens", "400,1200", "--label", "x"])
    assert rc == 0
    return out


def test_collect_sends_sequential_greedy_requests(tmp_path):
    mod, server = _tool(), _Server()
    try:
        out = _collect(tmp_path, mod, server)
    finally:
        server.httpd.shutdown()
    rep = json.loads(out.read_text())
    assert rep["label"] == "x" and rep["model"] == "stub-model" and rep["url"] == server.url
    items = rep["items"]
    assert [i["id"] for i in items] == ["prose#0@400", "prose#0@1200", "prose#1@400", "prose#1@1200",
                                        "code#0@400", "code#0@1200", "code#1@400", "code#1@1200"]
    first = items[0]
    assert {"id", "type", "max_tokens", "prompt", "text", "completion_tokens", "prompt_tokens",
            "finish_reason"} <= set(first)
    assert first["prompt"] == "prose p0" and first["text"] == "echo:prose p0:400 tail of the answer"
    assert first["completion_tokens"] == 5 and first["prompt_tokens"] == 7 and first["finish_reason"] == "length"
    assert server.most == 1                                   # one request at a time: every forward is one stream
    paths = {p for p, _ in server.bodies}
    assert paths == {"/v1/chat/completions"}
    for _, body in server.bodies:
        assert body["temperature"] == 0 and body["stream"] is False and body["model"] == "stub-model"
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["messages"][0]["role"] == "user"
    assert sorted(b["max_tokens"] for _, b in server.bodies) == [400] * 4 + [1200] * 4


def test_compare(tmp_path, capsys):
    mod, server = _tool(), _Server()
    try:
        a = _collect(tmp_path, mod, server)
    finally:
        server.httpd.shutdown()
    assert mod.main(["compare", str(a), str(a)]) == 0
    assert "identical 8/8" in capsys.readouterr().out
    rep = json.loads(a.read_text())
    t = rep["items"][3]["text"]
    rep["items"][3]["text"] = t[:10] + ("X" if t[10] != "X" else "Y") + t[11:]
    b = tmp_path / "b.json"
    b.write_text(json.dumps(rep))
    assert mod.main(["compare", str(a), str(b)]) == 1
    out = capsys.readouterr().out
    assert "prose#1@1200" in out and "char 10" in out and "identical 7/8" in out
    rep["items"] = rep["items"][:-1]
    rep["items"][3]["text"] = t
    b.write_text(json.dumps(rep))
    assert mod.main(["compare", str(a), str(b)]) == 1
    out = capsys.readouterr().out
    assert "missing" in out and "code#1@1200" in out
