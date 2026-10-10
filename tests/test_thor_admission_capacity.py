"""Admission control and GET /v1/capacity: deterministic, no GPU.

NoRoom is now a CapacityError (HTTP 503 everywhere that already checks isinstance(exc, CapacityError), with no
new code at those call sites); MemoryGate gains a reservation ledger so admission can refuse a new request before
it starts rather than squeezing an already-running one later; capacity_status.of() reads that ledger the same way
/health reads the decoder (health.py's own pattern), for a router or harness to poll before sending.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

GIB = 1 << 30


def _gate_mod():
    return importlib.import_module("tensorfold.cuda.memory_gate")


def test_noroom_is_a_capacity_error_with_a_reason_and_a_retry_hint():
    gate = _gate_mod()
    errors = importlib.import_module("tensorfold.server.errors")
    exc = gate.NoRoom("a prompt waits for memory", reason="memory", retry_after_s=5.0)
    assert isinstance(exc, errors.CapacityError) and isinstance(exc, errors.RequestError)
    assert exc.reason == "memory" and exc.retry_after_s == 5.0
    assert str(exc) == "a prompt waits for memory"


def test_noroom_defaults_reason_to_memory_and_retry_to_none():
    exc = _gate_mod().NoRoom("streams already wait for memory; a new request waits until one finishes")
    assert exc.reason == "memory" and exc.retry_after_s is None


def test_memory_gate_reservation_ledger_admits_until_room_is_promised():
    gate = _gate_mod().MemoryGate(10 * GIB, reserve=0)
    assert gate.reserved_total == 0 and gate.admits(10 * GIB)
    gate.promise(1, 6 * GIB)
    assert gate.reserved_total == 6 * GIB
    assert gate.admits(4 * GIB) and not gate.admits(4 * GIB + 1)
    gate.promise(2, 4 * GIB)
    assert not gate.admits(1)                                  # fully promised: a third reservation must wait
    assert gate.unpromise(1) == 6 * GIB and gate.reserved_total == 4 * GIB
    assert gate.admits(6 * GIB) and not gate.admits(6 * GIB + 1)


def test_unpromise_of_an_unknown_key_is_a_safe_no_op():
    gate = _gate_mod().MemoryGate(1 * GIB, reserve=0)
    assert gate.unpromise(999) == 0 and gate.reserved_total == 0


def _fake_app(*, live=0, prefilling=0, max_streams=8, window=262144, window_bytes=4 * GIB,
             room=20 * GIB, reserved=None, held=0, evicted_kept=0, kept=()):
    gate_mod = _gate_mod()
    gate = gate_mod.MemoryGate(room, reserve=0)
    gate.held = held
    gate.evicted_kept = evicted_kept
    for key, bytes_ in (reserved or {}).items():
        gate.promise(key, bytes_)
    decoder = SimpleNamespace(streams={i: object() for i in range(live)}, filling=[object()] * prefilling,
                              capacity=window, window_bytes=window_bytes, memory_gate=gate, kept=kept)
    scheduler = SimpleNamespace(decoder=decoder, max_streams=max_streams)
    return SimpleNamespace(engine=SimpleNamespace(scheduler=scheduler))


def test_capacity_status_reports_streams_and_free_cache_room():
    cap = importlib.import_module("tensorfold.cuda.capacity_status")
    app = _fake_app(live=2, prefilling=1, max_streams=8, window=262144, window_bytes=4 * GIB,
                    room=20 * GIB, reserved={1: 6 * GIB})
    body = cap.of(app)
    assert body["ok"] and body["live_streams"] == 2 and body["prefilling_streams"] == 1 and body["max_streams"] == 8
    assert body["context_window_tokens"] == 262144
    assert body["reserved_bytes"] == 6 * GIB and body["free_cache_bytes"] == 14 * GIB
    # 4 GiB for a full 262144-token window: 14 GiB of free room is ~917,504 tokens' worth, clamped to the window
    assert body["largest_admissible_prompt_tokens_estimate"] == 262144
    assert body["overloaded"] is False


def test_capacity_status_flags_overloaded_at_max_streams_or_no_free_room():
    cap = importlib.import_module("tensorfold.cuda.capacity_status")
    at_max = cap.of(_fake_app(live=8, max_streams=8, room=20 * GIB))
    assert at_max["overloaded"] is True
    full_room = cap.of(_fake_app(live=1, max_streams=8, room=4 * GIB, reserved={1: 4 * GIB}))
    assert full_room["free_cache_bytes"] == 0 and full_room["overloaded"] is True


def test_capacity_status_reports_kept_session_sizes_and_eviction_count():
    cap = importlib.import_module("tensorfold.cuda.capacity_status")
    kept = [([1] * 33000, None, {}, None), ([1] * 1600, None, {}, None)]
    body = cap.of(_fake_app(kept=kept, evicted_kept=4))
    assert sorted(k["tokens"] for k in body["kept_sessions"]) == [1600, 33000]
    assert body["kept_sessions_total_tokens"] == 34600
    assert body["evicted_kept_sessions_total"] == 4


def test_capacity_status_without_a_concurrent_decoder_is_a_clean_not_ok():
    cap = importlib.import_module("tensorfold.cuda.capacity_status")
    body = cap.of(SimpleNamespace(engine=SimpleNamespace(scheduler=None)))
    assert body["ok"] is False and "reason" in body
