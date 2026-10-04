"""Flash Next's decode host path on Jetson Thor (sm_110) (DEC-2, FR-OPS-17).

On by default on sm_110; ``TENSORFOLD_THOR_DEC2_HOST=0`` keeps the earlier path (the kill switch). What it changes:

- At load, after the startup read has put the PLE n-gram tables in the page cache, their pages are also mapped into the
  process (``MADV_POPULATE_READ``, ``host_table._populate``). A decode round gathers each stream's window rows from the
  memory-mapped bf16 table under ``MADV_RANDOM``; unmapped, a row's page costs a minor fault on the round's host path,
  where no forward runs (DEC-2 s2/s3). Page tables only: the pages stay clean, shared and evictable, the bytes read
  are the same, and a locked table (``lock``, mlock) is mapped already.
- ``stage`` gathers every stream's window rows in one call a table, split over the table's thread pool in pieces of
  ``TENSORFOLD_THOR_DEC2_SPLIT`` rows or more (default 32; ``=0``: the pages mapped only). The same rows, copied by
  more cores at once: with the pages mapped, a decode gather is the row copy's memory latency (DEC-2 s3).
"""

from __future__ import annotations

import os
import time

import torch

ENV = "TENSORFOLD_THOR_DEC2_HOST"
SPLIT_ENV = "TENSORFOLD_THOR_DEC2_SPLIT"
CAPABILITY = (11, 0)


class S:
    batch = False                  # stage: one gather a table for every stream (set at load with the pages mapped)


def available(device: int = 0) -> bool:
    """sm_110 and not switched off."""

    if os.environ.get(ENV, "1").strip().lower() in ("0", "off", "false", "no"):
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


def map_tables(tables) -> str:
    """Map every table's cached pages (``populate``); the startup line's note, with the page-table and memory cost."""

    before, began = _status(), time.perf_counter()
    try:
        tables = list(tables)
        size = sum(table.populate() for table in tables)
    except (AttributeError, OSError) as exc:          # a kernel before 5.14, or a table with nothing mapped
        return f"n-gram table pages not mapped ({type(exc).__name__}: {exc})"
    after = _status()
    split = int(os.environ.get(SPLIT_ENV, "32") or 0)
    tables = list(tables)
    if split > 0:
        for table in tables:
            table.split = split
        S.batch = True

    def grew(key: str, unit: int) -> float:
        return (after.get(key, 0) - before.get(key, 0)) / unit

    return (f"n-gram table pages mapped: {size / 2**30:.2f} GiB in {time.perf_counter() - began:.1f}s, VmPTE "
            f"+{grew('VmPTE', 1024):.1f} MiB, RssFile +{grew('RssFile', 1 << 20):.2f} GiB, MemAvailable "
            f"{before.get('MemAvailable', 0) / 2**20:.2f} -> {after.get('MemAvailable', 0) / 2**20:.2f} GiB; rows of "
            f"{getattr(tables[0], 'width', '?')} bf16; gathers " + (f"batched, pool pieces >= {split} rows" if S.batch else "as before"))
