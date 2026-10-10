"""Thor's device-fed MTP draft chain (DPUSH-6, ``thor_mtp_fed.chain``): with steps whose rows depend on their own inputs
only, the speculative chain (each step launched before the host reads the step before it) and the ``sync`` chain draft
what the host-fed chain drafts, leave each stream's MTP length where it leaves it, feed each row the token its own
stream picked, and never overwrite a step's picks before the host read them; 1, 4 and 8 streams, random rooms."""

import random

import pytest

pytest.importorskip("triton")

from tensorfold.families.qwen4_exp.cuda import thor_mtp_fed  # noqa: E402

DEPTH = 6


class _Stream:
    def __init__(self, sid: int) -> None:
        self.sid, self.drafts = sid, []


def _pick(seed: int, sid: int, j: int, token):
    """A row's pick from its own inputs only (stream, step, the token it was fed)."""

    r = random.Random(hash((seed, sid, j, -1 if token is None else token)))
    return r.randrange(50), r.random()


def _host(streams, room, confidence, seed):
    """ae716c1's host-fed loop (multi.py ``_draft_all``) on the same picks: drafts and each stream's chain steps."""

    steps = {s.sid: 0 for s in streams}
    active = list(streams)
    got = [_pick(seed, s.sid, 0, None) for s in streams]
    for j in range(DEPTH):
        nxt = []
        for s, (d, p) in zip(active, got):
            low = confidence > 0 and p < confidence
            if low and j > 0:
                continue
            s.drafts.append(d)
            if not low and j + 1 < room[s.sid]:
                nxt.append((s, d))
        if not nxt:
            return steps
        got = [_pick(seed, s.sid, j + 1, d) for s, d in nxt]
        for s, _ in nxt:
            steps[s.sid] += 1
        active = [s for s, _ in nxt]
    return steps


class _Device:
    """Two picks buffers by parity; a launched step's row r gathers the token row ``i`` of the step before picked."""

    def __init__(self, streams, live, seed):
        self.seed = seed
        self.picks = {0: [_pick(seed, s.sid, 0, None) for s in streams], 1: []}
        self.sids = {0: [s.sid for s in streams], 1: []}
        self.srows = {0: [r for _, _, r in live], 1: []}
        self.depth = {0: 0, 1: -1}
        self.unread = {0: True, 1: False}
        self.steps = {s.sid: 0 for s in streams}
        self.rows_launched = self.wasted = 0

    def launch(self, cands, par):
        assert cands, "a step of no rows launched"
        assert not self.unread[par], "a step's picks overwritten before the host read them"
        before = 1 - par
        picks, sids = [], []
        for s, i, r in cands:
            assert self.sids[before][i] == s.sid, "a row gathered another stream's pick"
            assert self.srows[before][i] == r, "a row gathered another row's streams"
            picks.append(_pick(self.seed, s.sid, self.depth[before] + 1, self.picks[before][i][0]))
            sids.append(s.sid)
            self.steps[s.sid] += 1
        self.picks[par], self.sids[par], self.srows[par] = picks, sids, list(range(len(cands)))
        self.depth[par], self.unread[par] = self.depth[before] + 1, True
        self.rows_launched += len(cands)
        return [s for s, _, _ in cands]

    def read(self, par, rows):
        assert self.unread[par] and len(self.picks[par]) == rows
        self.unread[par] = False
        return self.picks[par]

    def revert(self, s):
        self.steps[s.sid] -= 1
        self.wasted += 1


@pytest.mark.parametrize("streams", [1, 4, 8])
@pytest.mark.parametrize("confidence", [0.0, 0.3, 0.7])
def test_fed_chains_draft_what_the_host_fed_chain_drafts(streams, confidence):
    rng = random.Random(streams * 100 + int(confidence * 10))
    wasted = launched = 0
    for trial in range(300):
        seed = rng.randrange(1 << 30)
        room = {sid: rng.randint(1, DEPTH) for sid in range(streams)}
        ref = [_Stream(sid) for sid in range(streams)]
        steps = _host(ref, room, confidence, seed)
        for spec in (True, False):
            got = [_Stream(sid) for sid in range(streams)]
            live = [(s, i, 3 * i + 2) for i, s in enumerate(got)]
            dev = _Device(got, live, seed)
            took = []
            thor_mtp_fed.chain(live, room, DEPTH, confidence, dev.launch, dev.read, dev.revert, spec,
                               after=took.append)
            assert [s.drafts for s in got] == [s.drafts for s in ref], (trial, spec)
            assert dev.steps == steps, (trial, spec)
            assert all(not dev.unread[p] or dev.picks[p] == [] or spec for p in (0, 1))
            if spec:
                wasted, launched = wasted + dev.wasted, launched + dev.rows_launched
            else:
                assert dev.wasted == 0
    assert launched > 0 and (confidence > 0 or wasted == 0)


def test_mode_is_off_without_thor(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import thor_graphs

    monkeypatch.setattr(thor_mtp_fed._S, "mode", None)
    monkeypatch.setattr(thor_graphs, "available", lambda device=0: False)
    assert thor_mtp_fed.mode() == "0"
    monkeypatch.setattr(thor_mtp_fed._S, "mode", None)
    monkeypatch.setattr(thor_graphs, "available", lambda device=0: True)
    for value, want in (("0", "0"), ("off", "0"), ("1", "1"), ("sync", "sync"), ("check", "check"), ("x", "0")):
        monkeypatch.setenv(thor_mtp_fed.ENV, value)
        monkeypatch.setattr(thor_mtp_fed._S, "mode", None)
        assert thor_mtp_fed.mode() == want
    monkeypatch.delenv(thor_mtp_fed.ENV)
    monkeypatch.setattr(thor_mtp_fed._S, "mode", None)
    assert thor_mtp_fed.mode() == "1"
    monkeypatch.setattr(thor_mtp_fed._S, "mode", None)
