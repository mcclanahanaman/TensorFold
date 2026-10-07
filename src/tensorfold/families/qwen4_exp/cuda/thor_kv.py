"""Jetson Thor (sm_110) serving memory for Flash Next's concurrent decoder: the n-gram tables locked, one cache pool.

On by default on sm_110 with ``--parallel``; ``TENSORFOLD_THOR_KV_POOL=0`` keeps the earlier path (the kill switch:
streams' caches grow from the driver, the wide-piece pool guard, the tables' pages evictable).

Thor's GPU allocates from the host's memory. The mapped n-gram tables (26.8 GiB for the NVFP4 checkpoint) are page
cache, so ``MemAvailable`` counts them as free: a cache grow sized against it took their pages, and once the driver's
freed pages ran out the grow waited on the kernel's reclaim and compaction (FR-OPS-17 memd-census1: a 1.4 GiB grow took
70 s and left 9% of the tables on disk; the prompt after it ran at 476 tok/s, a ~2.6k tok/s prompt with them resident).
This path budgets the tables instead:

1. **Tables locked** (``mlock``) before the decoder measures its room: locked pages leave ``MemAvailable`` and no
   allocation can take them, so every lookup stays a memory read.
2. **One cache pool**: after the warm-ups, the room left past the reserve (``TENSORFOLD_MEMORY_RESERVE_GIB``: host
   memory for request decoding, the kernel and the driver) and the on-demand graphs' budget is allocated once and freed
   into torch's caching allocator. Later grows split that block instead of asking the driver for pages: no reclaim or
   compaction while serving. The memory gate's room is that pool less the kept snapshots' share, so admission and the
   kept-prompt policy work against memory that is really there.

Neither step changes what a request computes: the same caches in the same layout, from a different allocator block.
"""

from __future__ import annotations

import os

import torch

ENV = "TENSORFOLD_THOR_KV_POOL"
CAPABILITY = (11, 0)
GIB = float(1 << 30)
ALIGN = 2 << 20                  # the pool's size, a whole number of the allocator's 2 MiB rounding
SLACK = 256 << 20                # the gate's reserve in the pool: a resize's allocator rounding and small transients


def _on() -> bool:
    return os.environ.get(ENV, "1").strip().lower() not in ("0", "off", "false", "no")


def available(device: int = 0) -> bool:
    """sm_110 on the host's memory (an integrated GPU) and not switched off."""

    if not _on() or not torch.cuda.is_available():
        return False
    from tensorfold.cuda.capacity import unified

    return tuple(torch.cuda.get_device_capability(device)) == CAPABILITY and unified(torch)


def lock_tables(tables, read: bool) -> tuple[bool, str]:
    """Lock every table's pages in memory (reading them first unless ``read``): whether all locked, and the note."""

    from .thor_dec2 import _status

    tables = list(tables)
    size = sum(int(getattr(t, "nbytes", 0)) for t in tables)
    before = _status()
    done = []
    for table in tables:
        if not read:
            table.prefetch()
        if not table.lock():
            for t in done:                       # all or nothing: an unlocked table is the earlier path's
                _unlock(t)
            return False, (f"n-gram tables not locked (the memory-lock limit refused {size / GIB:.1f} GiB; "
                           "raise `ulimit -l`, or the tables' pages stay evictable)")
        done.append(table)
    after = _status()
    return True, (f"n-gram tables locked in memory: {size / GIB:.2f} GiB, MemAvailable "
                  f"{before.get('MemAvailable', 0) / 2**20:.2f} -> {after.get('MemAvailable', 0) / 2**20:.2f} GiB")


def _unlock(table) -> None:
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    libc.munlock.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    for arr in getattr(table, "values", []):
        libc.munlock(arr.ctypes.data, arr.nbytes)


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
        del block                                # torch's allocator keeps the segment: later grows split it
        return size
    return 0


__all__ = ["ENV", "available", "lock_tables", "reserve"]
