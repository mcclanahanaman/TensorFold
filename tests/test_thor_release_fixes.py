"""The Thor release re-gate fixes of admission and /v1/capacity (2026-10-08): deterministic, no GPU.

The cross-family review of the release's memory, admission and capacity code (VERDICT: FAIL) found: B1, the warm-up
request's admission reservation is never released; S1, GET /v1/capacity counts the Thor pool's reserve as free room
admission never promises; S2, three admission refusals carry no retry hint, so their 503s lack Retry-After; L1, a lane
yield leaves the yielded stream's held DeltaNet rows; L3, a prompt that leaves no reply room is a plain ValueError (a
500); L4, a kept point dropped without its whole chain is not counted. The release test list also failed on
test_flashnext_prefix_fork's refusal case: admission started a prompt's Thor NVMe arrival reads before its last
refusal point. Each test here fails on the frozen commit 746fe698 and passes with its fix.
"""

from __future__ import annotations

import importlib
import queue
from types import SimpleNamespace

import pytest

from tensorfold.cuda.streams import Stream
from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the module imports)
from tests.test_cuda_growing_caches import decoder, weights

GIB = 1 << 30


def mods():
    import torch

    return (importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi"),
            importlib.import_module("tensorfold.families.qwen4_exp.cuda.state"), torch)


def slot(state):
    return state.State(weights(), 256, 4, "bf16", limit=65536)


def admitting(multi, state, torch, room, monkeypatch=None):
    """A decoder whose ``admit`` runs for real (the ledger, the slot, the grow) up to the engine it would build."""

    dec = decoder(multi, state, torch, room=room)
    dec.capacity, dec.slots, dec.prefill_rows, dec.next_id = 65536, 8, 64, 0
    dec.points, dec.vision, dec.buf, dec.mbuf, dec.pbuf = None, None, None, None, None
    dec.evicted_by = {"memory": 0, "slots": 0, "entries": 0}
    dec._ref_state = slot(state)
    dec.window_bytes = dec._ref_state.cache_bytes(dec.capacity)
    for layer in dec.w.layers:
        layer.ple = None
    if monkeypatch is not None:                      # no engine: the prompt's prefill is not what these tests run
        monkeypatch.setattr(multi, "_slot", lambda *a, **k: SimpleNamespace())
        monkeypatch.setattr(multi, "prefill_begin", lambda *a, **k: 0)
        monkeypatch.setattr(multi.image_rows, "begin", lambda *a, **k: None)
    return dec


def app_of(dec, max_streams=8):
    return SimpleNamespace(engine=SimpleNamespace(scheduler=SimpleNamespace(decoder=dec, max_streams=max_streams)))


# -- B1: the warm-up's reservation -----------------------------------------------------------------------------------

def test_warm_releases_the_warm_up_requests_reservation_and_held_rows(allocations, monkeypatch):  # noqa: F811
    multi, state, torch = mods()
    dec = admitting(multi, state, torch, 1 << 40, monkeypatch)
    dec.free = [slot(state)]
    alive = []

    def one_round():                                 # the warm prompt fills, then it decodes a round, not yet done
        for s in list(dec.filling):
            dec.filling.remove(s)
            dec.fills.pop(s.sid)
            dec.streams[s.sid] = s
            dec.held[s.sid] = [1, 2, 3]
            alive.append(s)                          # kept alive: its id() cannot be reused and hide a stale entry
        return []

    dec.round = one_round
    dec._prerecord_experts = lambda: None
    dec.warm()
    assert alive and dec.streams == {} and dec.filling == []
    assert dec.memory_gate.reserved == {}            # 746fe698: {id(warm request): its worst case}, never released
    assert dec.held == {}


def test_a_request_that_fits_right_after_warm_up_is_admitted(allocations, monkeypatch):  # noqa: F811
    multi, state, torch = mods()
    dec = admitting(multi, state, torch, 1 << 40, monkeypatch)
    dec.free = [slot(state)]
    alive = []

    def one_round():
        for s in list(dec.filling):
            dec.filling.remove(s)
            dec.fills.pop(s.sid)
            dec.streams[s.sid] = s
            alive.append(s)
        return []

    dec.round = one_round
    dec._prerecord_experts = lambda: None
    dec.warm()
    dec.memory_gate.room = dec._worst(dec._ref_state, 100, 50)    # the room is exactly this request's worst case
    s = Stream([1] * 100, 50)
    dec.admit(s)                                                   # 746fe698: NoRoom (the warm-up's stale promise)
    assert dec.filling == [s] and dec.memory_gate.reserved_total == dec.memory_gate.room


# -- S1: GET /v1/capacity counts against the bound admission uses -----------------------------------------------------

def test_get_capacity_subtracts_the_pool_reserve_exactly_as_admission_does():
    from tests.test_thor_admission_capacity import _fake_app

    cap = importlib.import_module("tensorfold.cuda.capacity_status")
    app = _fake_app(room=10 * GIB, reserved={1: 6 * GIB})
    gate = app.engine.scheduler.decoder.memory_gate
    gate.pooled, gate.reserve = True, GIB            # Thor's KV pool: every grow keeps the reserve free
    body = cap.of(app)
    assert body["admission_reserve_bytes"] == GIB
    assert body["free_cache_bytes"] == 3 * GIB       # 746fe698: 4 GiB, the reserve counted as promisable
    assert gate.admits(body["free_cache_bytes"]) and not gate.admits(body["free_cache_bytes"] + 1)
    gate.promise(2, 3 * GIB)
    body = cap.of(app)
    assert body["free_cache_bytes"] == 0 and body["overloaded"] is True   # 746fe698: 1 GiB free, not overloaded
    gate.pooled = False                              # no pool: admission promises the whole room, as before
    body = cap.of(app)
    assert body["admission_reserve_bytes"] == 0 and body["free_cache_bytes"] == GIB


def test_get_capacity_reports_the_window_the_server_accepts():
    from tests.test_thor_admission_capacity import _fake_app

    cap = importlib.import_module("tensorfold.cuda.capacity_status")
    app = _fake_app(window=262151, window_bytes=4 * GIB)     # the decoder's rows: --context 262144 plus 7 draft rows
    assert cap.of(app)["context_window_tokens"] == 262151    # an app that does not say what it accepts: as before
    app.effective_context_window = 262144                    # the CUDA server refuses a prompt plus reply past this
    body = cap.of(app)
    assert body["context_window_tokens"] == 262144           # c4e0335: 262151, a request the server refuses as too long
    assert body["largest_request_tokens"] == 262144 and body["largest_admissible_prompt_tokens_estimate"] == 262144


def test_get_and_post_capacity_agree_at_the_admission_boundary(allocations):  # noqa: F811
    multi, state, torch = mods()
    cap = importlib.import_module("tensorfold.cuda.capacity_status")
    dec = admitting(multi, state, torch, 0)
    ref, gate = dec._ref_state, dec.memory_gate
    gate.room = dec._worst(ref, 60000, 0)
    gate.pooled = True                               # a reserve of one 8,192-row step (caches and a layer's copy)
    gate.reserve = dec._worst(ref, 16000, 0) - dec._worst(ref, 8000, 0)
    gate.promise(1, dec._worst(ref, 20000, 0))                      # a live request's promise
    dec.streams = {1: SimpleNamespace(st=slot(state), waiting=False, done=False)}
    body = cap.of(app_of(dec))
    free = body["free_cache_bytes"]
    prompt = [1] * 1000
    lo, hi = 1, dec.capacity - len(prompt) - dec.depth - 1
    while lo < hi:                                   # the largest reply whose worst case GET says still fits
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if dec.preview(prompt, mid)["need_bytes"] <= free else (lo, mid - 1)
    at, above = dec.preview(prompt, lo), dec.preview(prompt, lo + 1)
    assert at["need_bytes"] <= free < above["need_bytes"]
    assert at["admissible_now"] is True              # 746fe698: False (GET's free room included the reserve)
    assert above["admissible_now"] is False and above["reason"] == "memory"


def test_the_largest_request_and_the_length_refusal_use_the_admission_bound(allocations):  # noqa: F811
    multi, state, torch = mods()
    errors = importlib.import_module("tensorfold.server.errors")
    dec = admitting(multi, state, torch, 0)
    ref, gate = dec._ref_state, dec.memory_gate
    gate.room = dec._worst(ref, 20000, 0)                           # 24,576 rows' caches and one layer's copy
    gate.pooled, gate.reserve = True, gate.room - dec._worst(ref, 16000, 0)   # admission promises 16,384 rows' worth
    assert dec.room_tokens() == 16384 - dec.depth - 2               # 746fe698: 24,571 (the reserve included)
    with pytest.raises(errors.ContextLengthError, match=r"maximum context length is 16379 tokens"):
        dec.admit(Stream([1] * 18000, 1))                           # 746fe698: NoRoom "memory", refused forever
    assert gate.reserved_total == 0
    assert dec.preview([1] * 18000, 1)["reason"] == "length"       # 746fe698: "memory"


# -- S2: every admission refusal carries its retry hint ---------------------------------------------------------------

def test_every_noroom_admission_raises_carries_the_memory_retry_hint(allocations, monkeypatch):  # noqa: F811
    multi, state, torch = mods()
    gate_mod = importlib.import_module("tensorfold.cuda.memory_gate")
    sched = importlib.import_module("tensorfold.cuda.scheduler")
    dec = admitting(multi, state, torch, 1 << 40)
    dec.streams = {0: SimpleNamespace(st=slot(state), waiting=True, done=False)}
    with pytest.raises(gate_mod.NoRoom) as waiting:                    # a live stream already waits for memory
        dec.admit(Stream([1] * 10, 4))
    dec.streams[0].waiting = False
    dec.free = [slot(state)]
    dec._grow = lambda st, rows, **kw: False
    with pytest.raises(gate_mod.NoRoom) as grow:                       # its caches cannot grow while others live
        dec.admit(Stream([1] * 10, 4))
    assert dec.memory_gate.reserved_total == 0 and len(dec.free) == 1
    del dec._grow
    dec.memory_gate.room = 0
    monkeypatch.setattr(multi.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(multi.torch.cuda, "memory_allocated", lambda *a: 0)
    monkeypatch.setattr(multi, "cuda_limit_bytes", lambda: 1)
    with pytest.raises(gate_mod.NoRoom) as limit:                      # alone, past the explicit CUDA limit
        dec._grow(slot(state), 10000, alone=True)
    for exc in (waiting.value, grow.value, limit.value):              # 746fe698: retry_after_s None on all three
        assert exc.reason == "memory" and exc.retry_after_s == 5.0
    assert gate_mod.MEMORY_RETRY_S == 5.0 and gate_mod.SLOTS_RETRY_S == 2.0     # one definition, beside NoRoom
    assert (sched.MEMORY_RETRY_S, sched.SLOTS_RETRY_S) == (5.0, 2.0)            # the scheduler's names still answer


@pytest.mark.parametrize("reason, header", [("memory", "5"), ("slots", "2")])
@pytest.mark.parametrize("stream", [False, True])
def test_a_refusal_without_its_own_hint_still_answers_retry_after(tmp_path, reason, header, stream):
    pytest.importorskip("jinja2")
    import json

    from tensorfold.cuda.memory_gate import NoRoom
    from tests.test_cuda_admission_wait import CHAT, Gate, _post, _serve

    class Bare(Gate):
        def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, admitted=None):
            raise NoRoom("a prompt waits for memory until a live stream finishes", reason=reason)

    with _serve(tmp_path, Bare(wait=30.0, refuse=True)) as port:
        status, headers, text = _post(port, {**CHAT, "stream": stream})
    assert status == 503 and headers.get("Retry-After") == header     # 746fe698: no Retry-After header
    assert json.loads(text)["error"]["reason"] == reason


# -- L1, L3, L4 -------------------------------------------------------------------------------------------------------

def test_a_lane_yield_drops_the_yielded_streams_held_rows(allocations):  # noqa: F811
    multi, state, torch = mods()
    sched_mod = importlib.import_module("tensorfold.cuda.scheduler")
    dec = admitting(multi, state, torch, 1 << 40)
    bg = Stream([1, 2, 3], 10, background=True)
    bg.sid, bg.st = 5, slot(state)
    dec.streams, dec.held = {5: bg}, {5: [1, 2]}
    dec.memory_gate.promise(id(bg), 100)
    sched = object.__new__(sched_mod.Scheduler)
    sched.decoder, sched.max_streams, sched.yields = dec, 1, 0
    sched.waiting, sched.boxes = sched_mod.Waiting(), {id(bg): queue.Queue()}
    sched.waiting.put((Stream([4], 2), queue.Queue()))                 # a foreground request waits for the lane
    sched._yield()
    assert sched.yields == 1 and dec.streams == {} and dec.memory_gate.reserved_total == 0
    assert dec.held == {}                            # 746fe698: {5: [1, 2]}, one entry a yield for the server's life


def test_a_prompt_that_leaves_no_reply_room_is_a_context_length_refusal(allocations):  # noqa: F811
    multi, state, torch = mods()
    errors = importlib.import_module("tensorfold.server.errors")
    dec = admitting(multi, state, torch, 1 << 40)
    window = dec.capacity - dec.depth - 1
    with pytest.raises(errors.ContextLengthError, match=rf"maximum context length is {window} tokens"):
        dec.admit(Stream([1] * window, 4))           # 746fe698: a plain ValueError, which HTTP served as a 500
    assert dec.memory_gate.reserved_total == 0


def test_kept_points_dropped_without_their_chain_are_counted_and_reported(allocations):  # noqa: F811
    multi, state, torch = mods()
    prefixes = importlib.import_module("tensorfold.families.qwen4_exp.cuda.prefixes")
    cap = importlib.import_module("tensorfold.cuda.capacity_status")
    dec = admitting(multi, state, torch, 1 << 40)
    a, b = slot(state), slot(state)
    dec.streams = {1: SimpleNamespace(st=a, waiting=False, done=False), 2: SimpleNamespace(st=b, waiting=False,
                                                                                            done=False)}
    dec.keep, dec.kept = 2, [([1] * 10, a, {}, None), ([1] * 20, a, {}, None)]
    prefixes.remember(dec, [2] * 10, b, {}, None)     # past keep: a's inner point [1]*10 goes first
    assert dec.points_dropped == 1
    dec.keep = 1
    prefixes.remember(dec, [2] * 30, b, {}, None)     # both slots busy, no inner point left: the oldest point goes
    assert [len(k[0]) for k in dec.kept] == [30] and dec.points_dropped == 3
    assert dec.evicted_by == {"memory": 0, "slots": 0, "entries": 0}   # no chain was freed: not a chain eviction
    body = cap.of(app_of(dec))
    assert body["kept_points_dropped_total"] == 3 and body["evicted_kept_sessions_total"] == 0


# -- the release test list's failure: no arrival reads for a refused admission --------------------------------------

def reads_recorded(dec):
    reads = []
    table = SimpleNamespace(arrival_rows=64, ahead=lambda ids: reads.append([int(x) for x in ids]) or 1)
    ngram = SimpleNamespace(n=3, initial_history=lambda: None, ids=lambda history, tokens: tokens)
    dec.w.layers[0].ple = SimpleNamespace(table=table, ngram=ngram)
    return reads


def test_a_refused_admission_starts_no_nvme_arrival_reads(allocations, monkeypatch):  # noqa: F811
    multi, state, torch = mods()
    gate_mod = importlib.import_module("tensorfold.cuda.memory_gate")
    dec = admitting(multi, state, torch, 1 << 40, monkeypatch)
    reads = reads_recorded(dec)
    dec.streams = {0: SimpleNamespace(st=slot(state), waiting=False, done=False)}
    spare = slot(state)
    dec.free = [spare]
    dec._grow = lambda st, rows, **kw: False         # held for memory: the scheduler retries it every pass
    with pytest.raises(gate_mod.NoRoom):
        dec.admit(Stream(list(range(100)), 8))
    assert reads == [] and dec.free == [spare]       # 746fe698: the refused request had started its reads
    dec._grow = lambda st, rows, **kw: True
    dec.admit(Stream(list(range(100)), 8))
    assert reads == [list(range(64))]                # an admitted one still reads its first rows from admission on
