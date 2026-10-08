"""The CUDA scheduler's bounded admission wait (``TENSORFOLD_ADMISSION_WAIT_S``; par-1-fix, FR-OPS-17) and its HTTP
receipts: a request that has waited that long for a seat or for cache room is refused with NoRoom ("slots" or
"memory", with a retry hint), which the chat route answers as HTTP 503 + Retry-After + reason before any stream opens;
a request that does not accept a refusal, or any request without the setting, waits its turn as before. CPU fakes."""

import http.client
import json
import threading
import time
from types import SimpleNamespace

import pytest

from tensorfold.cuda.memory_gate import NoRoom
from tensorfold.cuda.scheduler import WAIT_ENV, Scheduler, admission_wait_s


class Seats:
    """A decoder whose streams each take one token a round once ``go`` is set; ``room=False``: a request that arrives
    while another is live is held for memory (NoRoom), as the Flash Next decoder's admission does."""

    def __init__(self, room: bool = True) -> None:
        self.streams: dict = {}
        self.filling: list = []
        self.room = room
        self.go = threading.Event()
        self.next = 0

    def live(self):
        return len(self.streams) + len(self.filling)

    def admit(self, s):
        if not self.room and self.streams:
            raise NoRoom("no room until a live stream ends", reason="memory")
        s.sid, self.next = self.next, self.next + 1
        self.streams[s.sid] = s

    def round(self):
        if not self.go.wait(0.01):
            return []
        done = []
        for s in list(self.streams.values()):
            s.take([7])
            if s.done:
                done.append(s)
        return done

    def finish(self, done):
        for s in done:
            self.streams.pop(s.sid, None)

    def drop(self):
        out = list(self.streams.values())
        self.streams.clear()
        return out


def _until(check, seconds=5.0):
    end = time.monotonic() + seconds
    while not check():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.005)


def _first(scheduler, events, name="a"):
    """Submit one request on a thread, admitted at once; returns the thread (its stats land in ``events``)."""

    def go():
        events.append((name, scheduler.submit([1], 1, None, True, lambda new: None,
                                              admitted=lambda: events.append(f"{name}-admitted"))))

    t = threading.Thread(target=go, daemon=True)
    t.start()
    _until(lambda: f"{name}-admitted" in events)
    return t


def test_a_request_that_waits_past_the_bound_for_a_seat_is_refused_with_slots():
    decoder = Seats()
    scheduler = Scheduler(decoder, max_streams=1, max_wait_s=0.2)
    try:
        events: list = []
        a = _first(scheduler, events)
        t0 = time.monotonic()
        with pytest.raises(NoRoom) as info:
            scheduler.submit([2], 1, None, True, lambda new: None, admitted=lambda: events.append("b-admitted"))
        waited = time.monotonic() - t0
        assert info.value.reason == "slots" and info.value.retry_after_s == 2.0
        assert 0.2 <= waited < 2.0
        assert "b-admitted" not in events
        assert scheduler.refused == {"slots": 1, "memory": 0}
        decoder.go.set()
        a.join(5)
        assert events[-1][0] == "a" and events[-1][1]["rounds"] == 0      # the live request ended normally
    finally:
        decoder.go.set()
        scheduler.close()


def test_a_request_held_for_memory_past_the_bound_is_refused_with_memory():
    decoder = Seats(room=False)
    scheduler = Scheduler(decoder, max_streams=4, max_wait_s=0.2)
    try:
        events: list = []
        a = _first(scheduler, events)
        with pytest.raises(NoRoom) as info:
            scheduler.submit([2], 1, None, True, lambda new: None, admitted=lambda: events.append("b-admitted"))
        assert info.value.reason == "memory" and info.value.retry_after_s == 5.0
        assert scheduler.refused == {"slots": 0, "memory": 1} and scheduler.held is None
        decoder.go.set()
        a.join(5)
        assert not a.is_alive()
    finally:
        decoder.go.set()
        scheduler.close()


def test_a_request_without_admitted_waits_its_turn_and_is_never_refused():
    decoder = Seats()
    scheduler = Scheduler(decoder, max_streams=1, max_wait_s=0.1)
    try:
        events: list = []
        a = _first(scheduler, events)
        b = threading.Thread(target=lambda: events.append(("b", scheduler.submit([2], 1, None, True,
                                                                                lambda new: None))), daemon=True)
        b.start()
        time.sleep(0.5)                                  # five times the bound: still waiting, not refused
        assert b.is_alive() and scheduler.refused == {"slots": 0, "memory": 0}
        decoder.go.set()
        a.join(5)
        b.join(5)
        assert [e[0] for e in events if isinstance(e, tuple)] == ["a", "b"]
    finally:
        decoder.go.set()
        scheduler.close()


def test_a_zero_wait_admits_a_request_a_free_seat_takes_and_refuses_one_at_once_when_full():
    decoder = Seats()
    scheduler = Scheduler(decoder, max_streams=1, max_wait_s=0)
    try:
        events: list = []
        a = _first(scheduler, events)                   # a free seat: admitted, never refused
        t0 = time.monotonic()
        with pytest.raises(NoRoom) as info:
            scheduler.submit([2], 1, None, True, lambda new: None, admitted=lambda: None)
        assert info.value.reason == "slots" and time.monotonic() - t0 < 1.0
        decoder.go.set()
        a.join(5)
        assert scheduler.refused == {"slots": 1, "memory": 0}
    finally:
        decoder.go.set()
        scheduler.close()


def test_admitted_runs_when_the_seat_is_taken_before_any_token():
    decoder = Seats()
    decoder.go.set()
    scheduler = Scheduler(decoder, max_streams=2, max_wait_s=30)
    try:
        order: list = []
        stats = scheduler.submit([1], 3, None, True, lambda new: order.append(("tokens", list(new))),
                                 admitted=lambda: order.append("admitted"))
        assert order[0] == "admitted" and order[1:] == [("tokens", [7])] * 3
        assert stats["rounds"] == 0
    finally:
        scheduler.close()


def test_the_wait_comes_from_the_environment_and_is_unbounded_when_unset(monkeypatch):
    monkeypatch.delenv(WAIT_ENV, raising=False)
    assert admission_wait_s() is None
    decoder = Seats()
    scheduler = Scheduler(decoder)
    try:
        assert scheduler.max_wait_s is None
    finally:
        scheduler.close()
    monkeypatch.setenv(WAIT_ENV, "12.5")
    assert admission_wait_s() == 12.5
    monkeypatch.setenv(WAIT_ENV, "0")
    assert admission_wait_s() == 0.0
    for bad in ("-1", "soon", "nan", "inf"):
        monkeypatch.setenv(WAIT_ENV, bad)
        with pytest.raises(ValueError, match=WAIT_ENV):
            admission_wait_s()


# -- HTTP: the chat route's receipts ---------------------------------------------------------------------------------

pytest.importorskip("jinja2")


class Gate:
    """A Flash-Next-like engine: ``refuse`` raises the scheduler's refusal before the seat, else it takes its seat
    (``admitted``) and answers "ok". Its scheduler's bounded wait is ``wait`` (None: unbounded, today's path)."""

    def __init__(self, wait, refuse):
        from tests.test_cuda_tool_choice import EOS
        self.eos = (EOS,)
        self.scheduler = SimpleNamespace(max_wait_s=wait)
        self.refuse = refuse
        self.calls: list = []

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, admitted=None):
        self.calls.append(admitted is not None)
        if self.refuse:
            raise NoRoom("this server is at capacity", reason="slots", retry_after_s=2.0)
        if admitted is not None:
            admitted()
        on_tokens([ord("o"), ord("k")])
        return {"rounds": 1}


def _post(port, body):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read().decode()
    finally:
        connection.close()


def _serve(tmp_path, engine):
    from tests.test_cuda_admission import http_server
    from tests.test_token_routes import cuda_app
    app = cuda_app(tmp_path)
    app.engine = engine
    return http_server(app)


CHAT = {"model": "fake-cuda", "max_tokens": 4, "messages": [{"role": "user", "content": "Hi"}]}


@pytest.mark.parametrize("stream", [True, False])
def test_a_bounded_wait_s_refusal_is_a_503_with_retry_after_and_reason_before_any_stream(tmp_path, stream):
    engine = Gate(wait=30.0, refuse=True)
    with _serve(tmp_path, engine) as port:
        status, headers, text = _post(port, {**CHAT, "stream": stream})
    assert status == 503 and headers.get("Retry-After") == "2"
    error = json.loads(text)["error"]
    assert error["reason"] == "slots" and "capacity" in error["message"]
    assert engine.calls == [True]                        # the request accepted a refusal


def test_a_bounded_wait_s_admitted_stream_opens_at_its_seat_and_answers(tmp_path):
    engine = Gate(wait=30.0, refuse=False)
    with _serve(tmp_path, engine) as port:
        status, headers, text = _post(port, {**CHAT, "stream": True})
    assert status == 200 and headers.get("Content-Type") == "text/event-stream"
    frames = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]
    assert frames[0]["choices"][0]["delta"] == {"role": "assistant"}
    deltas = [f["choices"][0]["delta"] for f in frames if f.get("choices")]
    assert "".join(d.get("reasoning_content", "") + d.get("content", "") for d in deltas) == "ok"   # thinking on
    assert text.rstrip().endswith("data: [DONE]")


def test_without_a_bounded_wait_a_stream_opens_at_once_as_before(tmp_path):
    engine = Gate(wait=None, refuse=True)
    with _serve(tmp_path, engine) as port:
        status, headers, text = _post(port, {**CHAT, "stream": True})
    assert status == 200                                 # today's path: the refusal is the open stream's error event
    errors = [json.loads(line[6:])["error"] for line in text.splitlines()
              if line.startswith("data: {") and "error" in json.loads(line[6:])]
    assert errors and errors[0]["reason"] == "slots"
    assert engine.calls == [False]
