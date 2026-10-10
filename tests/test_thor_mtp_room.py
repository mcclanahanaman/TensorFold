"""The multi-stream decoder's draft chain at the end of a stream's caches (rcfix-1: HumanEval+ HTTP 500s, "MTP context
past the cache capacity"). A prompt that joins a round from its pass has the caches admission sized (prompt + depth + 2
rows), while its first chain may need up to prompt + 2 x depth MTP rows; the chain then stopped mid-way with an error
that failed every request of the round. Each stream's chain now ends inside its caches, drafting what a chain with
unbounded caches drafts wherever that chain stayed inside them (and its first drafts where it did not), and a stream
whose MTP step still finds no room fails alone while the others draft as they would without it. 1 and 8 streams, the
host-fed chain and the device-fed one (speculative and ``sync``), with and without the confidence stop."""

import importlib
import random
from types import SimpleNamespace

import pytest

from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the modules import)

pytestmark = pytest.mark.torch
torch = pytest.importorskip("torch")

DEPTH, WIDTH, BIG = 6, 4, 1 << 20
CHAINS = ["0", "1", "sync"]                  # host-fed; device-fed speculative; device-fed in the host's order


def _pick(seed: int, sid: int, row: int, token: int):
    """A draft and its probability from the step's own inputs: the stream, its MTP row and the token fed to it."""

    r = random.Random(hash((seed, sid, row, token)))
    return r.randrange(1, 50), r.random()


class _Event:
    def synchronize(self) -> None:
        pass

    def record(self) -> None:
        pass


class _State:
    def __init__(self, pos: int, mtp_len: int, drafted: int, capacity: int) -> None:
        self.pos, self.mtp_len, self.mtp_drafted, self.capacity, self.limit = pos, mtp_len, drafted, capacity, BIG
        self.most = 0                               # the most MTP rows any step of this round left

    def set_mtp_len(self, n: int) -> None:
        self.mtp_len = n
        self.most = max(self.most, n)


def _buffers(rows: int):
    return SimpleNamespace(staged=_Event(), ids_host=torch.zeros(rows, dtype=torch.int32),
                           ids=torch.zeros(rows, dtype=torch.int32), last_host=torch.zeros(rows, dtype=torch.int32),
                           last=torch.zeros(rows, dtype=torch.int32), mtp_in=torch.zeros(rows, WIDTH),
                           streams=torch.zeros(rows, WIDTH))


class _Head:
    """The MTP head's GPU work, faked: each step's segment picks ``_pick`` of its stream, its last row and that row's
    token (``mtp_stage`` / ``_fed_launch`` wrote it to the buffers' ids). ``fail``: (state, step) whose MTP step raises
    ``MtpRoomError`` once, before the step changes any length (as the stage's check does)."""

    def __init__(self, multi, dec, seed: int) -> None:
        self.multi, self.dec, self.seed = multi, dec, seed
        self.sids, self.steps, self.fail = {}, [], None

    def __call__(self, segs) -> list:
        if self.fail is not None and len(self.steps) == self.fail[1] and any(st is self.fail[0] for st, _, _ in segs):
            st, self.fail = self.fail[0], None
            raise self.multi.MtpRoomError(st)
        out = []
        for st, a0, a1 in segs:
            row = st.mtp_len + (a1 - a0) - 1          # read before the caller bumps the stream's MTP length
            assert row < st.capacity, "an MTP step past the stream's caches"
            out.append(_pick(self.seed, self.sids[id(st)], row, int(self.dec.mbuf.ids[a1 - 1])))
        self.steps.append([self.sids[id(st)] for st, _, _ in segs])
        return out


def _decoder(multi, chain: str, confidence: float, seed: int, slots: int):
    dec = multi.MultiDecoder.__new__(multi.MultiDecoder)
    rows = slots * (DEPTH + 1)
    dec.depth, dec.confidence, dec.graphs, dec.kept, dec.fed, dec.fed_check = DEPTH, confidence, None, [], chain, None
    dec.w, dec.buf, dec.mbuf = None, _buffers(rows), _buffers(rows)
    head = dec.head = _Head(multi, dec, seed)
    dec._mtp = lambda segs, k=None: head(segs)
    dec._picks = lambda logits, positions, samplings, k=None: logits
    if chain != "0":
        dec.fed_tok, dec.fed_ar = torch.zeros(slots, dtype=torch.int32), torch.arange(slots, dtype=torch.int32)
        dec.fed_host, dec.fed_dev = torch.zeros(2 * slots, dtype=torch.int32), torch.zeros(2 * slots, dtype=torch.int32)
        dec.fed_staged, pending = _Event(), {}

        def step(segs, par):                          # the picks kernel: each row's draft id feeds the next step
            picks = head(segs)
            dec.fed_tok[:len(picks)] = torch.tensor([d for d, _ in picks], dtype=torch.int32)
            pending[par] = picks

        dec._fed_step = step
        dec._fed_read = lambda par, rows: pending[par][:rows]
    return dec


def _stream(sid: int, keep: int, base: int, drafted: int, cap: int, left: int):
    """A stream after its round's commit: ``keep`` kept rows past ``base`` MTP rows, the last chain's ``drafted`` rows
    still on top, ``left`` reply tokens to go after this round's, caches of ``cap`` rows."""

    from tensorfold.cuda.streams import Stream

    s = Stream([1] * 8, 3 + keep + left)
    s.sid, s.out = sid, [7] * 3
    s.st = _State(base + keep, base + drafted, drafted, cap)
    return s


def _random(rng, n: int, tight: float) -> list:
    """``n`` streams as (stream, its first row in the round's buffers, its kept rows' next tokens); with probability
    ``tight`` its caches end 0..DEPTH+1 rows past its kept rows (a prompt that joined from its pass)."""

    out = []
    for sid in range(n):
        keep, base = rng.randint(1, DEPTH + 1), rng.randint(1, 4000)
        cap = base + keep + rng.randint(0, DEPTH + 1) if rng.random() < tight else BIG
        drafted = min(rng.randint(0, DEPTH - 1), cap - base)
        left = rng.choice([DEPTH + 40, rng.randint(1, DEPTH)])
        out.append((_stream(sid, keep, base, drafted, cap, left), sid * (DEPTH + 1),
                    [rng.randrange(1, 50) for _ in range(keep)]))
    return out


def _copy(streams, cap=None) -> list:
    out = []
    for s, a0, keep in streams:
        st = s.st
        c = _stream(s.sid, len(keep), st.mtp_len - st.mtp_drafted, st.mtp_drafted, st.capacity if cap is None else cap,
                    s.count - len(s.out) - len(keep))
        out.append((c, a0, list(keep)))
    return out


def _run(multi, chain, confidence, seed, streams, *, old=False, fail=None):
    """``_draft_all`` (``old``: without this fix's room bound, as a28a67b) on ``streams``; returns the decoder."""

    dec = _decoder(multi, chain, confidence, seed, max((a0 for _, a0, _ in streams), default=0) // (DEPTH + 1) + 1)
    for s, _, _ in streams:
        dec.head.sids[id(s.st)] = s.sid
    if fail is not None:
        dec.head.fail = (streams[fail[0]][0].st, fail[1])
    if not old:
        dec._draft_all(streams, {})
        return dec
    for s, _, _ in streams:
        s.drafts = []
    room = {s.sid: min(DEPTH, s.count - len(s.out) - len(keep)) for s, _, keep in streams}
    todo = [t for t in streams if room[t[0].sid] > 0]
    for s, _, _ in todo:
        s.st.set_mtp_len(s.st.mtp_len - s.st.mtp_drafted)
        s.st.mtp_drafted = 0
    if todo:
        dec._draft_chain(todo, room, {})
    return dec


def _result(streams) -> list:
    return [(s.sid, list(s.drafts), s.st.mtp_len, s.st.mtp_drafted, s.error is not None) for s, _, _ in streams]


@pytest.mark.parametrize("streams", [1, 8])
@pytest.mark.parametrize("chain", CHAINS)
@pytest.mark.parametrize("confidence", [0.0, 0.6])
def test_a_chain_ends_inside_its_caches_and_drafts_what_unbounded_caches_draft(allocations, streams, chain, confidence):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    rng = random.Random(streams * 1000 + CHAINS.index(chain) * 10 + int(confidence * 10))
    passed = 0
    for trial in range(250):
        seed = rng.randrange(1 << 30)
        start = _random(rng, streams, 0.5)
        free = _copy(start, cap=BIG)                  # the same round with unbounded caches: the reference
        _run(multi, chain, confidence, seed, free)
        got = _copy(start)
        _run(multi, chain, confidence, seed, got)
        past = False
        for (s, _, keep), (r, _, _) in zip(got, free):
            st, cap, base = s.st, s.st.capacity, r.st.mtp_len - r.st.mtp_drafted - len(keep)
            assert s.error is None and st.most <= cap, trial      # (speculative steps included)
            if r.st.mtp_len <= cap:                  # inside the caches: the same drafts and MTP rows as before
                assert (s.drafts, st.mtp_len, st.mtp_drafted) == (r.drafts, r.st.mtp_len, r.st.mtp_drafted), trial
            else:                                    # past them: the first drafts, up to the caches' last row
                past = True
                assert s.drafts == r.drafts[:cap - base - len(keep) + 1] and st.mtp_len <= cap, trial
        if past:                                     # a28a67b's chain on the same round: the 500 (the whole round)
            passed += 1
            with pytest.raises(multi.MtpRoomError, match="MTP context past the cache capacity"):
                _run(multi, chain, confidence, seed, _copy(start), old=True)
        else:
            old = _copy(start)
            _run(multi, chain, confidence, seed, old, old=True)
            assert _result(old) == _result(got), trial
    assert passed >= 3                              # rounds where the bound mattered (the old chain failed)


@pytest.mark.parametrize("chain", CHAINS)
def test_a_prompt_that_joined_its_round_near_its_first_rows_drafts_inside_them(allocations, chain):  # noqa: F811
    """The HumanEval+ case: a 248-token prompt whose admission kept the slot's first 256 rows; its first round kept all
    seven rows (the token and six drafts) and its chain had six drafts of room, past row 256 at its third step."""

    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    joined = _stream(0, DEPTH + 1, 248, DEPTH - 1, 256, 500)
    others = [_stream(sid, 3, 3000 + sid, 2, 8192, 500) for sid in range(1, 8)]
    start = [(joined, 0, [5] * (DEPTH + 1))] + [(s, sid * (DEPTH + 1), [9, 8, 7]) for sid, s in enumerate(others, 1)]
    with pytest.raises(multi.MtpRoomError):
        _run(multi, chain, 0.0, 1, _copy(start), old=True)
    got, free = _copy(start), _copy(start, cap=BIG)
    _run(multi, chain, 0.0, 1, got)
    _run(multi, chain, 0.0, 1, free)
    assert got[0][0].drafts == free[0][0].drafts[:2] and got[0][0].st.mtp_len == 256
    assert _result(got)[1:] == _result(free)[1:]
    assert not any(s.error for s, _, _ in got)


@pytest.mark.parametrize("streams", [1, 8])
@pytest.mark.parametrize("chain", CHAINS)
@pytest.mark.parametrize("confidence", [0.0, 0.6])
def test_a_stream_without_room_fails_alone_and_the_others_draft_as_without_it(allocations, streams, chain, confidence):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    rng = random.Random(7 + streams * 100 + CHAINS.index(chain) * 10 + int(confidence * 10))
    for trial in range(150):
        seed = rng.randrange(1 << 30)
        start = _random(rng, streams, 0.3)
        v = rng.randrange(streams)
        probe = _copy(start)
        dec = _run(multi, chain, confidence, seed, probe)
        steps = [j for j, sids in enumerate(dec.head.steps) if v in sids]
        if trial % 3 == 0:                           # its kept rows do not fit (its first step)
            vs = start[v][0].st
            vs.capacity = vs.mtp_len - vs.mtp_drafted + len(start[v][2]) - 1
            fail = None
        else:                                        # an MTP step mid-chain finds no room
            fail = (v, rng.choice(steps))
        got = _copy(start)
        _run(multi, chain, confidence, seed, got, fail=fail)
        alone = _copy([t for t in start if t[0].sid != v])
        _run(multi, chain, confidence, seed, alone)
        victim = got[v][0]
        assert isinstance(victim.error, multi.MtpRoomError) and victim.drafts == [], trial
        assert (victim.st.mtp_len, victim.st.mtp_drafted) == (start[v][0].st.mtp_len - start[v][0].st.mtp_drafted, 0)
        assert [r for r in _result(got) if r[0] != v] == _result(alone), trial
