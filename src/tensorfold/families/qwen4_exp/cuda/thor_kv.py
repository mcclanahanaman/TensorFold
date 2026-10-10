"""Jetson Thor (sm_110) serving memory for Flash Next's concurrent decoder: the n-gram tables locked, one cache pool.

On by default on sm_110 with ``--parallel``; ``TENSORFOLD_THOR_KV_POOL=0`` keeps the earlier path (the kill switch:
streams' caches grow from the driver, the wide-piece pool guard, the tables' pages evictable).

Thor's GPU allocates from the host's memory. The mapped n-gram tables (26.8 GiB for the NVFP4 checkpoint) are page
cache, so ``MemAvailable`` counts them as free: a cache grow sized against it took their pages, and once the driver's
freed pages ran out the grow waited on the kernel's reclaim and compaction (a 1.4 GiB grow took
70 s and left 9% of the tables on disk; the prompt after it ran at 476 tok/s, a ~2.6k tok/s prompt with them resident).
This path budgets the tables instead:

1. **Tables locked** (``mlock``) before the decoder measures its room: locked pages leave ``MemAvailable`` and no
   allocation can take them, so every lookup stays a memory read.
2. **One cache pool**: after the warm-ups, the room left past the reserve (``TENSORFOLD_MEMORY_RESERVE_GIB``: host
   memory for request decoding, the kernel and the driver) and the on-demand graphs' budget is allocated once and freed
   into torch's caching allocator. Later grows split that block instead of asking the driver for pages: no reclaim or
   compaction while serving. The memory gate's room is that pool less the kept snapshots' share, so admission and the
   kept-prompt policy work against memory that is really there.
3. **The idle on its design point** (``fit``): ``MemAvailable`` foretells the pool's cost only roughly (the page cache,
   the driver's freed-page pool and the kernel's watermarks move with a 28 GiB allocation; the idle
   came out 1.05 GiB under the reserve on one start, and 1.6 GiB over it on another whose reservation swapped the
   startup's heap out). After the reservation the host's free memory is read again and the pool re-reserved by the
   miss, so every start idles at, never under, the reserve plus the on-demand graphs' budget plus the vision host
   buffers. Swap the reservation filled counts as used where it is on disk (it comes back at the first touch); a
   swapped-out page still in RAM (the swap cache) is already out of ``MemAvailable`` and is not counted twice
   (vmem-a5: 725-751 MiB of the 1.6 GiB stayed in RAM). The server's own init after the engine (the tokenizer, the
   chat template) is landed too: the CLI re-reads the idle once it is built (``MultiDecoder.refit_pool``).
   ``TENSORFOLD_THOR_KV_POOL_FIT=0`` keeps the one sizing from ``MemAvailable`` before the reservation.

Neither step changes what a request computes: the same caches in the same layout, from a different allocator block.
"""

from __future__ import annotations

import os

import torch

ENV = "TENSORFOLD_THOR_KV_POOL"
TOUCH_ENV = "TENSORFOLD_THOR_KV_POOL_TOUCH"     # diagnostics: =0 leaves the pool's pages untouched until first use
CAPABILITY = (11, 0)
GIB = float(1 << 30)
ALIGN = 2 << 20                  # the pool's size, a whole number of the allocator's 2 MiB rounding
SLACK = 256 << 20                # the gate's reserve in the pool: a resize's allocator rounding and small transients
FIT_ENV = "TENSORFOLD_THOR_KV_POOL_FIT"     # =0: the pool sized once, from MemAvailable before it (the earlier sizing)
FIT_TOL = 64 << 20               # how far above its design point the idle may land (never below) before a re-reserve
FIT_AIM = FIT_TOL // 2           # a re-reservation aims the idle at the middle of that window
FIT_TRIES = 3                    # re-reservations at most (each one ~1-2 s at startup)


def _on() -> bool:
    return os.environ.get(ENV, "1").strip().lower() not in ("0", "off", "false", "no")


def fit_on() -> bool:
    """The fit re-reserves (``FIT_ENV`` unset or on); off, the pool is sized once and only measured."""
    return os.environ.get(FIT_ENV, "1").strip().lower() not in ("0", "off", "false", "no")


def available(device: int = 0) -> bool:
    """sm_110 on the host's memory (an integrated GPU) and not switched off."""

    if not _on() or not torch.cuda.is_available():
        return False
    from tensorfold.cuda.capacity import unified

    return tuple(torch.cuda.get_device_capability(device)) == CAPABILITY and unified(torch)


def _arrays(table) -> list:
    """A host table's mapped arrays, whatever its format (bf16 or FP8 values, NVFP4 codes and scales, MLX words)."""

    return [a for name in ("values", "scales", "words", "biases") for a in (getattr(table, name, None) or [])]


def lock_tables(tables, read: bool) -> tuple[bool, str]:
    """Lock every table's pages in memory (reading them first unless ``read``): whether all locked, and the note.
    All or nothing: on a refusal what was locked is unlocked again (the earlier path's evictable pages)."""

    import ctypes

    from .thor_dec2 import _status

    tables = list(tables)
    arrays = [a for t in tables for a in _arrays(t)]
    size = sum(a.nbytes for a in arrays)
    if not arrays:
        return False, "n-gram tables not locked (no mapped arrays)"
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mlock.argtypes = libc.munlock.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    before = _status()
    if not read:
        for table in tables:
            table.prefetch()                     # eight readers first: mlock alone faults the pages in one by one
    done = []
    for arr in arrays:
        if libc.mlock(arr.ctypes.data, arr.nbytes) != 0:
            err = ctypes.get_errno()
            for a in done:
                libc.munlock(a.ctypes.data, a.nbytes)
            return False, (f"n-gram tables not locked ({os.strerror(err)} locking {size / GIB:.1f} GiB; "
                           "`ulimit -l` must allow it), their pages stay evictable")
        done.append(arr)
    after = _status()
    return True, (f"n-gram tables locked in memory: {size / GIB:.2f} GiB, MemAvailable "
                  f"{before.get('MemAvailable', 0) / 2**20:.2f} -> {after.get('MemAvailable', 0) / 2**20:.2f} GiB")


def reserve(size: int) -> int:
    """Allocate ``size`` bytes (rounded down to ``ALIGN``) once and free them into torch's cache: the bytes held.

    An allocation the driver refuses is retried 512 MiB smaller (the pool is what is really there, not a request)."""

    size = max(0, int(size)) // ALIGN * ALIGN
    while size > 0:
        try:
            block = torch.empty((size,), dtype=torch.uint8, device="cuda")
        except torch.OutOfMemoryError:
            size = max(0, size - (512 << 20)) // ALIGN * ALIGN
            continue
        if os.environ.get(TOUCH_ENV, "1").strip() not in ("0", "off", "false", "no"):
            block.zero_()                        # every page touched once now, not by the first requests' grows
            torch.cuda.synchronize()
        del block                                # torch's allocator keeps the segment: later grows split it
        return size
    return 0


def trim_host() -> None:
    """Before the pool measures its room: unreachable objects collected, then glibc's ``malloc_trim(0)``, so the
    load's freed host memory is back with the kernel and the room is measured on what the server really keeps."""

    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):          # not glibc: nothing to hand back this way
        pass


def _settled(read, *, every: float = 0.2, within: int = 8 << 20, limit: float = 3.0) -> dict | None:
    """``read()`` (``/proc/meminfo`` as bytes) once two reads ``every`` seconds apart agree on ``MemAvailable`` within
    ``within``, or the last read after ``limit`` seconds (the kernel's reclaim after a large allocation, then its boosted
    watermarks, settle in ~2 s on Thor). A read that fails (None) ends the wait on the last good one; None if the
    first fails."""

    import time

    last, stop = read(), time.monotonic() + limit
    while last is not None and time.monotonic() < stop:
        time.sleep(every)
        now = read()
        if now is None:
            break
        if abs(now["MemAvailable"] - last["MemAvailable"]) <= within:
            return now
        last = now
    return last


def swapped(before: dict, now: dict) -> int:
    """Swap filled since ``before`` that is on disk: ``SwapFree``'s drop less ``SwapCached``'s change. A page swapped
    out but still in RAM (the swap cache) is already out of ``MemAvailable``; counting its swap slot too would count
    it twice. On-disk swap is the used slots less the swap cache, so the change counts with its sign: a swap cache
    that shrank to disk adds to the drop (one that shrank by freeing its slots raised ``SwapFree`` as much)."""

    drop = int(before.get("SwapFree", 0)) - int(now.get("SwapFree", 0))
    cached = int(now.get("SwapCached", 0)) - int(before.get("SwapCached", 0))
    return max(0, drop - cached)


def fit(size: int, target: int, before: dict | None, read=None) -> tuple[int, list[dict]]:
    """Land the host's idle free memory on ``target`` after ``reserve(size)``: read ``MemAvailable`` once it settles and
    re-reserve the pool by the miss (smaller when the reservation took more than ``MemAvailable`` foretold, larger when
    it took less), at most ``FIT_TRIES`` times, until the idle is between ``target`` and ``FIT_TOL`` above it (a
    re-reservation aims at ``FIT_AIM`` above). Swap filled since ``before`` and on disk (``swapped``) counts as used:
    those pages come back at the first touch. ``read`` None (not a unified GPU), ``FIT_ENV=0`` or a failed first read
    measures at most once and keeps the pool.

    Returns the pool's bytes and one record a measurement: the pool then, MemAvailable, the swap filled, the idle."""

    steps: list[dict] = []
    if read is None or before is None:
        return size, steps
    tries = FIT_TRIES if fit_on() else 0
    for i in range(tries + 1):
        now = _settled(read)
        if now is None:                        # no reading: the pool stays as it is
            break
        gone = swapped(before, now)
        idle = int(now["MemAvailable"]) - gone
        steps.append({"pool": size, "available": int(now["MemAvailable"]), "swapped": gone, "idle": idle})
        over = idle - target                   # < 0: the pool took more than MemAvailable foretold
        new = size + over - FIT_AIM
        if 0 <= over <= FIT_TOL or i == tries or new <= 0:
            break
        torch.cuda.synchronize()
        torch.cuda.empty_cache()               # the pool's block back to the driver: nothing holds a piece of it yet
        size = reserve(new)
    return size, steps


__all__ = ["ENV", "FIT_ENV", "available", "fit", "fit_on", "lock_tables", "reserve", "swapped", "trim_host"]
