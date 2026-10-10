"""Flash Next's device-fed MTP draft chain on Jetson Thor (sm_110) (DPUSH-6, FR-OPS-17).

A round's draft chain is host-bound: each MTP step waited for the host to read the step before it (picks, stop rule),
stage the next window (tokens and positions) and launch the next graph. Here each step's input token comes from the
step before it on the device: the step's picks kernel also writes each row's draft id, and the next step gathers its
rows' ids and input streams from there. So step j+1 is launched for every stream that could still draft (it has
room) before the host reads step j's picks (``spec``); the host's stop rule is the one it always was (a step's low
pick after the first stops a stream and is dropped; the first step's is kept), and a stream that stops has its
launched step's length bump undone, so its one extra MTP entry sits past ``mtp_len`` (the next absorb writes those
positions, in order, before any row reads them: the same as a rejected draft's entries). The drafts are the host-fed
chain's because a row's bits depend on its own inputs only: the picks kernel reduces each row in its own program, in a
fixed order (``torch.logsumexp``'s order depends on the rows reduced together), and the step's other kernels are the
per-row ones the verification windows already rely on (``check`` measures it in the server, per round).

``TENSORFOLD_THOR_MTP_FED``: ``1`` (default on sm_110 with decode graphs on), ``0`` the kill switch (the host-fed chain
of ae716c1), ``sync`` (this chain without the speculation: each step launched after the stop rule, its survivors
only), ``check`` (diagnostics: each round runs the speculative chain, the ``sync`` chain and the host-fed chain on the
same inputs, compares their drafts, the picks' probability bits and the MTP lengths, and keeps the host-fed one; the
counts go to ``TENSORFOLD_THOR_MTP_FED_LOG`` or stderr at exit). Greedy streams only: a round with a sampled stream
drafts host-fed.
"""

from __future__ import annotations

import atexit
import json
import os
import sys

import torch
import triton
import triton.language as tl

ENV = "TENSORFOLD_THOR_MTP_FED"
LOG_ENV = "TENSORFOLD_THOR_MTP_FED_LOG"
MODES = ("0", "1", "sync", "check")
BLOCK = 2048                    # columns a picks program reads per pass
WARPS = 4
DUMP_EVERY = 200               # check mode: rounds between count dumps


class _S:
    mode: str | None = None


def mode() -> str:
    """This process's chain: ``1``, ``0``, ``sync`` or ``check`` (sm_110 with decode graphs on; else ``0``). Decided
    once."""

    if _S.mode is None:
        from . import thor_graphs

        value = os.environ.get(ENV, "1").strip().lower()
        value = {"off": "0", "false": "0", "no": "0", "on": "1", "true": "1", "yes": "1"}.get(value, value)
        _S.mode = value if value in MODES and thor_graphs.available() else "0"
    return _S.mode


@triton.jit
def _picks(X, stride, n, GOT, TOK, MAP, HAS_MAP: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = X + row.to(tl.int64) * stride
    offs = tl.arange(0, BLOCK)
    best = tl.full((BLOCK,), float("-inf"), tl.float32)
    for c0 in range(0, n, BLOCK):
        cols = c0 + offs
        best = tl.maximum(best, tl.load(base + cols, mask=cols < n, other=float("-inf")).to(tl.float32))
    top = tl.max(best, axis=0)
    first = tl.full((BLOCK,), n, tl.int32)
    acc = tl.zeros((BLOCK,), tl.float32)
    for c0 in range(0, n, BLOCK):
        cols = c0 + offs
        x = tl.load(base + cols, mask=cols < n, other=float("-inf")).to(tl.float32)
        first = tl.minimum(first, tl.where(x == top, cols, n))
        acc += tl.exp(x - top)
    col = tl.min(first, axis=0)
    lse = top + tl.log(tl.sum(acc, axis=0))
    tl.store(GOT + row * 3, top)
    tl.store(GOT + row * 3 + 1, col.to(tl.float32))
    tl.store(GOT + row * 3 + 2, lse)
    if HAS_MAP:
        tl.store(TOK + row, tl.load(MAP + col).to(tl.int32))
    else:
        tl.store(TOK + row, col)


def picks(logits: torch.Tensor, got: torch.Tensor, tok: torch.Tensor, idmap: torch.Tensor | None) -> torch.Tensor:
    """Each row's greedy pick (capturable): ``got[r] = [top logit, its first column, logsumexp]`` (float32) and
    ``tok[r]`` its draft id (``idmap[column]``, or the column). One program a row, its reductions in a fixed order: a
    row's bits do not depend on the rows beside it. Returns ``got[:rows]``."""

    rows, n = logits.shape
    if logits.stride(1) != 1:
        raise ValueError("picks: the logits' columns must be contiguous")
    _picks[(rows,)](logits, logits.stride(0), n, got, tok, tok if idmap is None else idmap,
                    HAS_MAP=idmap is not None, BLOCK=BLOCK, num_warps=WARPS)
    return got[:rows]


def chain(live: list, room: dict, depth: int, confidence: float, launch, read, revert, spec: bool,
          after=None, record: dict | None = None) -> None:
    """A round's draft chain after its first MTP step (the absorb, already launched into parity 0).

    ``live``: the first step's rows ``[(stream, picks row, streams row)]``; ``room``: stream id -> drafts it may take;
    ``launch(cands, parity)``: stage and launch an MTP step for ``cands`` ``[(stream, picks row, streams row)]`` (rows
    of the step before it) into ``parity``'s picks buffer, bump their lengths, return the streams in row order;
    ``read(parity, rows)``: that step's ``[(draft, p)]`` a row; ``revert(stream)``: undo a launched step's bump for a
    stream that stopped; ``after(took)``: host work while the next step runs. ``spec``: launch step j+1 for every
    stream with room before reading step j's picks (else after the stop rule, for its survivors: the host-fed order).
    ``record``: stream id -> every ``(draft, p)`` read for it (diagnostics)."""

    par, rows = 0, len(live)
    first = [x for x in live if 1 < room[x[0].sid]] if spec else []
    launched = launch(first, 1) if first else []         # (no stream with room for a second draft: nothing to launch)
    for j in range(depth):
        got = read(par, rows)
        nxt, took = [], []
        for s, i, _ in live:
            d, p = got[i]
            if record is not None:
                record.setdefault(s.sid, []).append((d, p))
            low = confidence > 0 and p < confidence
            if low and j > 0:
                continue
            s.drafts.append(d)
            took.append((s, d))
            if not low and j + 1 < room[s.sid]:
                nxt.append(s)
        if spec:
            going = {id(s) for s in nxt}
            for s in launched:
                if id(s) not in going:
                    revert(s)
            at = {id(s): i for i, s in enumerate(launched)}
            live = [(s, at[id(s)], at[id(s)]) for s in nxt]
        elif nxt:
            at = {id(s): (i, r) for s, i, r in live}
            launched = launch([(s,) + at[id(s)] for s in nxt], 1 - par)
            live = [(s, i, i) for i, s in enumerate(launched)]
        if not nxt:
            return
        par, rows = 1 - par, len(launched)
        if spec:
            cand = [x for x in live if j + 2 < room[x[0].sid]]
            launched = launch(cand, 1 - par) if cand else []
        if after is not None:
            after(took)


class Check:
    """``check`` mode's counts: rounds, picks compared, and the rounds whose speculative chain differs from the
    ``sync`` chain (drafts, probability bits or MTP lengths) or whose drafts differ from the host-fed chain's."""

    def __init__(self) -> None:
        self.n = {"rounds": 0, "streams": 0, "picks": 0, "spec_vs_sync": 0, "lens_vs_sync": 0, "drafts_vs_host": 0,
                  "lens_vs_host": 0, "p_vs_host": 0}
        self.first: list = []
        atexit.register(self.dump)

    def add(self, runs: dict, lens: dict) -> None:
        spec, sync, host = runs["spec"], runs["sync"], runs["host"]
        self.n["rounds"] += 1
        self.n["streams"] += len(spec)
        self.n["picks"] += sum(len(r) for _, r in spec.values())
        for sid in spec:
            bad = []
            if spec[sid] != sync[sid]:
                self.n["spec_vs_sync"] += 1
                bad.append("spec_vs_sync")
            if lens["spec"][sid] != lens["sync"][sid]:
                self.n["lens_vs_sync"] += 1
                bad.append("lens_vs_sync")
            if spec[sid][0] != host[sid][0]:
                self.n["drafts_vs_host"] += 1
                bad.append("drafts_vs_host")
            if lens["spec"][sid] != lens["host"][sid]:
                self.n["lens_vs_host"] += 1
                bad.append("lens_vs_host")
            if [d for d, _ in spec[sid][1]] == [d for d, _ in host[sid][1]] and \
                    any(abs(a - b) > 1e-5 for (_, a), (_, b) in zip(spec[sid][1], host[sid][1])):
                self.n["p_vs_host"] += 1
                bad.append("p_vs_host")
            if bad and len(self.first) < 16:
                self.first.append({"what": bad, "streams": len(spec), "spec": spec[sid], "sync": sync[sid],
                                   "host": host[sid], "lens": {k: v[sid] for k, v in lens.items()}})
        if self.n["rounds"] % DUMP_EVERY == 0:
            self.dump()

    def dump(self) -> None:
        line = json.dumps({"thor_mtp_fed_check": self.n, "first": self.first}, default=str)
        path = os.environ.get(LOG_ENV)
        if path:                                         # the latest counts (rewritten every DUMP_EVERY rounds)
            with open(path, "w", encoding="utf-8") as f:
                f.write(line + "\n")
        else:
            print(line, file=sys.stderr)
