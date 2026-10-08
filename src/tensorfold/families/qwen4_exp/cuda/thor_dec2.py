"""Flash Next's decode host path on Jetson Thor (sm_110) (DEC-2, FR-OPS-17).

On by default on sm_110; ``TENSORFOLD_THOR_DEC2_HOST=0`` keeps the earlier path (the kill switch). Each piece has its own
switch (all on by default) so an A/B can attribute it:

- At load, after the startup read has put the PLE n-gram tables in the page cache, their pages are also mapped into the
  process (``MADV_POPULATE_READ``, ``host_table._populate``). A decode round gathers each stream's window rows from the
  memory-mapped bf16 table under ``MADV_RANDOM``; unmapped, a row's page costs a minor fault on the round's host path,
  where no forward runs (DEC-2 s2/s3). Page tables only: the pages stay clean, shared and evictable, the bytes read
  are the same, and a locked table (``lock``, mlock) is mapped already.
- ``stage`` gathers every stream's window rows in one call a bf16 table. A call of ``TENSORFOLD_THOR_DEC2_POOL_ROWS``
  rows or more (default 384) is split over the table's pool in pieces of ``TENSORFOLD_THOR_DEC2_SPLIT`` rows or more
  (default 32; ``=0``: the pages mapped only); fewer rows, such as one stream's window, copy on the calling thread:
  there, waking the pool costs more than it saves (DEC-2 s3: one stream 0.996 with the pool).
- ``TENSORFOLD_THOR_DEC2_AHEAD``: while each MTP step runs on the GPU, the rows of the tokens the next window is known to
  start with (the round's last token, then each draft as it is picked) are gathered and kept on the stream's state;
  ``stage`` reuses them when the state's n-gram history is the same object and they are a prefix of the window, and
  gathers the rest. Exact: a token's row ids depend on the n-1 tokens before it (an EOS among them resets), so the ids
  after a prefix are the ids with the last n-1 tokens of history + prefix as history.
- ``TENSORFOLD_THOR_DEC2_TABLES``: the attention ``Step`` and DeltaNet ``Tables`` send their device tables from numpy
  (no Python lists), the DeltaNet conv and state pointers come from each state's bases and strides, and the attention
  cache pointers from an array a stream kept until its caches move. The same values.
- ``TENSORFOLD_THOR_NGRAM_PAIRS``: an NVFP4 table gathers through its pair table (``NVFP4Table.use_pairs``: one lookup
  a code byte instead of about ten row-sized temporaries a call, fresh host pages each time). The same bits.
- ``TENSORFOLD_THOR_DEC2_CHECK=1`` (diagnostics, off): every reused window's ids and every fast pointer table are checked
  against the earlier computation (the earlier one is used on a mismatch); the counts print at exit.
"""

from __future__ import annotations

import atexit
import os
import time

import numpy as np
import torch

ENV = "TENSORFOLD_THOR_DEC2_HOST"
SPLIT_ENV = "TENSORFOLD_THOR_DEC2_SPLIT"
POOL_ENV = "TENSORFOLD_THOR_DEC2_POOL_ROWS"
AHEAD_ENV = "TENSORFOLD_THOR_DEC2_AHEAD"
TABLES_ENV = "TENSORFOLD_THOR_DEC2_TABLES"
CHECK_ENV = "TENSORFOLD_THOR_DEC2_CHECK"
PAIRS_ENV = "TENSORFOLD_THOR_NGRAM_PAIRS"
CAPABILITY = (11, 0)
_NP = {torch.int32: np.int32, torch.int64: np.int64}


class S:
    batch = False                  # stage: one gather a bf16 table for every stream (set at load)
    split = 0                      # pool pieces of this many rows or more, for a call of ``pool_rows`` rows or more
    pool_rows = 384
    ahead = False                  # the next window's known rows gathered while the MTP steps run (set at load)
    tables = None                  # Step/Tables from numpy with kept pointers (decided at first use)
    check = False                  # diagnostics: reused ids and fast pointers checked against the earlier path
    stats = {"windows": 0, "reused": 0, "tokens": 0, "missed": 0, "mismatch": 0, "ptr_checks": 0, "ptr_mismatch": 0,
              "ptr_mismatch_attn": 0, "ptr_mismatch_gdn": 0}  # diagnostics only (par-1-fix): which table desynced


def _on(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "off", "false", "no")


def available(device: int = 0) -> bool:
    """sm_110 and not switched off."""

    if not _on(ENV):
        return False
    return torch.cuda.is_available() and torch.cuda.get_device_capability(device) == CAPABILITY


def _status() -> dict:
    """This process's VmRSS, RssFile and VmPTE and the system's MemAvailable, in KiB (Linux; empty elsewhere)."""

    got = {}
    for path, keys in (("/proc/self/status", ("VmRSS", "RssFile", "VmPTE")), ("/proc/meminfo", ("MemAvailable",))):
        try:
            with open(path) as f:
                for line in f:
                    key = line.split(":")[0]
                    if key in keys:
                        got[key] = int(line.split()[1])
        except OSError:
            pass
    return got


def map_tables(tables: list) -> tuple[bool, str]:
    """Map every table's cached pages (``populate``): whether it worked, and the note with its memory cost."""

    before, began = _status(), time.perf_counter()
    try:
        size = sum(table.populate() for table in tables)
    except (AttributeError, OSError) as exc:          # a kernel before 5.14, or a table with nothing mapped
        return False, f"n-gram table pages not mapped ({type(exc).__name__}: {exc})"
    after = _status()

    def grew(key: str, unit: int) -> float:
        return (after.get(key, 0) - before.get(key, 0)) / unit

    return True, (f"n-gram table pages mapped: {size / 2**30:.2f} GiB in {time.perf_counter() - began:.1f}s, VmPTE "
                  f"+{grew('VmPTE', 1024):.1f} MiB, RssFile +{grew('RssFile', 1 << 20):.2f} GiB, MemAvailable "
                  f"{before.get('MemAvailable', 0) / 2**20:.2f} -> {after.get('MemAvailable', 0) / 2**20:.2f} GiB; rows "
                  f"of {getattr(tables[0], 'width', '?') if tables else '?'} bf16")


def setup(tables, locked: bool) -> str:
    """At load: map the tables' pages (a locked table's are mapped), set the gather and ahead switches; the note."""

    tables = list(tables)
    if tables and all(getattr(t, "nvme", False) for t in tables):    # rows from NVMe: nothing to map, never a fault
        mapped, note = True, "n-gram rows from NVMe through the row cache"
    else:
        mapped, note = (True, "n-gram table pages locked") if locked else map_tables(tables)
    bf16 = bool(tables) and all(getattr(t, "bits", 4) == 16 for t in tables)
    S.split = max(0, int(os.environ.get(SPLIT_ENV, "32") or 0))
    S.pool_rows = max(1, int(os.environ.get(POOL_ENV, "384") or 384))
    S.batch = mapped and bf16 and S.split > 0
    S.ahead = bf16 and _on(AHEAD_ENV)
    S.check = _on(CHECK_ENV, "0")
    pairs = [t for t in tables if hasattr(t, "use_pairs")] if _on(PAIRS_ENV) else []
    for table in pairs:
        table.use_pairs()
    if S.check:
        atexit.register(lambda: print(f"[dec2] check {S.stats}", flush=True))
    pool = f"batched, pool pieces >= {S.split} rows from {S.pool_rows} rows" if S.batch else "as before"
    return (f"{note}; gathers {pool}; ahead {'on' if S.ahead else 'off'}; tables {'on' if tables_on() else 'off'}"
            + (f"; NVFP4 rows through the pair table ({PAIRS_ENV}=0: dequantized a call)" if pairs else "")
            + ("; CHECK" if S.check else ""))


def tables_on() -> bool:
    """Step/Tables from numpy with kept pointers (sm_110 and both switches on); decided once."""

    if S.tables is None:
        S.tables = available() and _on(TABLES_ENV)
    return S.tables


def ahead_on(w) -> bool:
    if not S.ahead or w.x3 is not None:
        return False
    from .forward import stage_ahead

    return stage_ahead()


def gather_rows(table, flat: np.ndarray) -> np.ndarray:
    """A bf16 table's rows for the ids ``flat``: pool pieces of ``S.split`` rows for a call of ``S.pool_rows`` or more."""

    if S.split > 0 and flat.size >= S.pool_rows:
        return table.gather(flat, S.split)
    return table.gather(flat)


def _ids_many(ngram, items: list) -> list:
    """``ngram.ids(history, tokens)`` for each item ([1+ tokens] or None for none); one-token items in one batch."""

    out: list = [None] * len(items)
    one = [i for i, (_, toks) in enumerate(items) if len(toks) == 1]
    if len(one) > 1 and len({len(items[i][0]) for i in one}) == 1:
        seq = np.stack([np.concatenate([np.asarray(items[i][0], dtype=np.int64), np.asarray(items[i][1], dtype=np.int64)])
                        for i in one])
        rows = ngram.rows_of(seq)[:, -1:]
        for j, i in enumerate(one):
            out[i] = rows[j]
    for i, (history, toks) in enumerate(items):
        if out[i] is None and len(toks):
            out[i] = ngram.ids(history, toks)
    return out


def ple_ahead(w, items: list, start: bool) -> None:
    """While an MTP step runs: the n-gram rows of ``items`` [(state, token)], each the next token a stream's next window
    is known to hold; ``start``: its first token (a new stash), else appended to the stream's stash."""

    if not items:
        return
    if start:
        for st, _ in items:
            st.dec2_ahead = {}
    for layer in w.layers:
        p = layer.ple
        if p is None:
            continue
        key, todo = id(p), []
        for st, tok in items:
            stash = getattr(st, "dec2_ahead", None)
            if stash is None:
                continue
            if start:
                entry = stash[key] = (st.ple_history, [], [st.ple_history], [], [])
            else:
                entry = stash.get(key)
                if entry is None or entry[0] is not st.ple_history:
                    continue
            todo.append((entry, int(tok)))
        if not todo:
            continue
        ids = _ids_many(p.ngram, [(entry[2][-1], (tok,)) for entry, tok in todo])
        heads = ids[0].size
        rows = gather_rows(p.table, np.concatenate([i.reshape(-1) for i in ids]))
        for j, ((hist0, toks, hists, idb, blocks), tok) in enumerate(todo):
            toks.append(tok)
            hists.append(np.concatenate([hists[-1], np.asarray([tok], dtype=np.int64)])[-len(hist0):])
            idb.append(ids[j])
            blocks.append(rows[j * heads:(j + 1) * heads])


def _take(st, p, toks: np.ndarray):
    """The rows gathered ahead for ``toks``' first tokens: (row blocks, id blocks, tokens, history after them) or None."""

    stash = getattr(st, "dec2_ahead", None)
    entry = stash.pop(id(p), None) if stash else None
    if entry is None:
        return None
    S.stats["windows"] += 1
    hist0, stoks, hists, idb, blocks = entry
    k = 0
    if hist0 is st.ple_history:
        while k < len(stoks) and k < len(toks) and stoks[k] == toks[k]:
            k += 1
    if k == 0:
        S.stats["missed"] += 1
        return None
    S.stats["reused"] += 1
    S.stats["tokens"] += k
    return blocks[:k], idb[:k], k, hists[k]


def stage_lookups(w, windows, segs) -> tuple[list, dict]:
    """``stage``'s lookups [(PLE, ids [rows, heads], first staging row)] with the rows gathered ahead as
    {lookup: (row blocks, tokens they cover)}; the rests' ids in one batch a table."""

    lookups, reuse = [], {}
    for layer in w.layers:
        p = layer.ple
        if p is None:
            continue
        todo = []                                   # (lookup, history, the window, ahead or None)
        for (st, tokens), (_, a0, _) in zip(windows, segs):
            toks = np.asarray(tokens, dtype=np.int64)
            got = _take(st, p, toks)
            todo.append((len(lookups), st.ple_history, toks, got))
            st.ple_last = (st.ple_history, toks)
            lookups.append([p, None, a0])
        rests = _ids_many(p.ngram, [(h if got is None else got[3], toks if got is None else toks[got[2]:])
                                    for _, h, toks, got in todo])
        for (i, h, toks, got), rest in zip(todo, rests):
            if got is None:
                ids = rest
            else:
                ids = np.concatenate(list(got[1]) + ([rest] if rest is not None else []))
                if S.check:
                    full = p.ngram.ids(h, toks)
                    if not np.array_equal(full, ids):
                        S.stats["mismatch"] += 1
                        ids, got = full, None
                if got is not None:
                    reuse[i] = (got[0], got[2])
            lookups[i][1] = ids
            lookups[i][2] = lookups[i][2] * ids.shape[-1]
    return [tuple(x) for x in lookups], reuse


def gather_lookups(lookups: list, reuse: dict) -> list:
    """Each lookup's rows: an array, or row blocks (``reuse``'s, then the rest's); with ``S.batch`` one call a bf16
    table for every stream's rest."""

    got: list = [None] * len(lookups)
    by: dict = {}
    for i, (p, ids, _) in enumerate(lookups):
        k = reuse[i][1] if i in reuse else 0
        rest = ids[k:]
        if rest.size == 0:
            got[i] = list(reuse[i][0])
        elif S.batch and getattr(p.table, "bits", 4) == 16:
            by.setdefault(id(p.table), (p.table, []))[1].append((i, rest.reshape(-1)))
        else:
            rows = p.table.gather(rest)
            got[i] = rows if i not in reuse else list(reuse[i][0]) + [rows]
    for table, items in by.values():
        rows = gather_rows(table, items[0][1] if len(items) == 1 else np.concatenate([r for _, r in items]))
        at = 0
        for i, r in items:
            part = rows[at:at + r.size]
            at += r.size
            got[i] = part if i not in reuse else list(reuse[i][0]) + [part]
    return got


def to_dev(values: np.ndarray, dtype: torch.dtype, device) -> torch.Tensor:
    """``gdn.to_device`` from numpy: the same values in one pinned, non-blocking copy, without a Python list."""

    arr = np.ascontiguousarray(np.asarray(values).reshape(-1), dtype=_NP[dtype])
    return torch.from_numpy(arr).pin_memory().to(device, non_blocking=True)


def attn_ptrs(st, layers) -> np.ndarray:
    """A stream's attention cache pointers [layers, 6] (keys, values, key scales, value scales, index keys, pooled),
    kept on its state until the caches move (``State.resize`` sets a new capacity; a clone has its own lists).

    The key uses ``st.version`` (``State.resize`` increments it; its own docstring: "a graph's pointer table
    refreshes on a change"), not a sampled pointer. With 8 concurrent streams sharing one caching allocator, a
    resize's freed block for one layer can be handed straight back by the allocator while a sibling layer's block
    goes to a different, concurrently-resizing stream (and vice versa): a key that samples only ``kc[0]``'s address
    can then match by coincidence while other layers' pointers have actually moved, serving a stale table (par-1-fix:
    measured 31/802 stale tables with TENSORFOLD_THOR_DEC2_CHECK=1 on a 92160-context 8-stream soak, all ``attn``,
    0 ``gdn``; ``gdn_ptrs`` has no cache to go stale). ``version`` is a monotonic per-state counter, so it cannot
    repeat by allocator coincidence."""

    key = (st.capacity, id(st.kc), id(st.ikc), id(st.pooled), len(layers), st.version)
    kept = getattr(st, "dec2_ptrs", None)
    if kept is not None and kept[0] == key:
        return kept[1]
    out = np.empty((len(layers), 6), dtype=np.int64)
    for i, layer in enumerate(layers):
        a = st.att_index[layer.index]
        kc, ikc, pooled = st.kc[a], st.ikc[a], st.pooled[a]
        out[i] = (kc.k.data_ptr(), kc.v.data_ptr(), kc.ks.data_ptr(), kc.vs.data_ptr(), ikc.data_ptr(), pooled.data_ptr())
    st.dec2_ptrs = (key, out)
    return out


def gdn_ptrs(st, lin: int) -> np.ndarray:
    """A stream's DeltaNet conv and state pointers [2, lin] (``st.conv[li]``, ``st.rec[st.cur[li], li]``) from the two
    buffers' bases and strides: the same addresses without a view a layer."""

    li = np.arange(lin, dtype=np.int64)
    conv, rec = st.conv, st.rec
    cur = np.asarray(st.cur[:lin], dtype=np.int64)
    return np.stack([conv.data_ptr() + li * (conv.stride(0) * conv.element_size()),
                     rec.data_ptr() + (cur * rec.stride(0) + li * rec.stride(1)) * rec.element_size()])


def same(fast: np.ndarray, slow: np.ndarray, kind: str = "") -> np.ndarray:
    """Diagnostics: a fast pointer table checked against the earlier one (which is used on a mismatch); ``kind``
    (``"attn"`` or ``"gdn"``, par-1-fix) breaks the mismatch count down by which table desynced."""

    S.stats["ptr_checks"] += 1
    if fast.shape != slow.shape or not np.array_equal(fast, slow):
        S.stats["ptr_mismatch"] += 1
        if kind:
            S.stats[f"ptr_mismatch_{kind}"] = S.stats.get(f"ptr_mismatch_{kind}", 0) + 1
        return slow
    return fast
