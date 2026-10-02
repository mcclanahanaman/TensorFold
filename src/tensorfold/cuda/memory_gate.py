"""Stream caches take memory as they grow: the room the startup plan left them, what they hold, and a reserve."""

from __future__ import annotations

from typing import Callable

from .capacity import cuda_limit_bytes


class NoRoom(RuntimeError):
    """A request that can't start until a live stream finishes and frees its caches."""


class MemoryGate:
    """``room`` bytes for every stream's growing caches; ``fits`` also asks the host (MemAvailable on a unified GPU)."""

    def __init__(self, room: int, reserve: int, live: Callable[[], int] | None = None,
                 pool: Callable[[], int] | None = None) -> None:
        self.room, self.reserve, self.held = int(room), int(reserve), 0
        self.live = live
        self.pool = pool                 # AGX Thor's wide prompt pieces: the GPU pages a grow takes without reclaim
        self.waits = self.ends = 0

    def fits(self, extra: int) -> bool:
        """Whether ``extra`` more bytes can be allocated now (a resize holds its old buffers until the copy lands)."""

        if self.held + int(extra) > self.room - self.reserve:
            return False
        return self.live is None or self.live() >= int(extra) + self.reserve

    def take(self, extra: int) -> None:
        self.held += int(extra)

    def give(self, freed: int) -> None:
        self.held = max(0, self.held - int(freed))

    def short(self, extra: int) -> bool:
        """Whether ``extra`` more bytes and the reserve exceed ``pool`` (False without one): past it, AGX Thor's GPU
        allocations reclaim and compact the page cache, the mapped n-gram tables, for seconds to minutes (ROWS-2)."""

        return self.pool is not None and self.pool() < int(extra) + self.reserve


def torch_live(torch, available: Callable) -> Callable[[], int]:
    """Reusable allocator bytes plus free memory, capped by the budget left after current allocations."""

    def room() -> int:
        allocated = int(torch.cuda.memory_allocated())
        free = int(available(torch)) + int(torch.cuda.memory_reserved()) - allocated
        limit = cuda_limit_bytes()
        return max(0, min(free, limit - allocated)) if limit is not None else max(0, free)

    return room


def _meminfo() -> dict | None:
    try:
        rows = open("/proc/meminfo").read().splitlines()
        return {key.rstrip(":"): int(value) * 1024 for key, value, *_ in (row.split() for row in rows)}
    except (OSError, ValueError):
        return None


def driver_pool(torch, meminfo: Callable[[], dict | None] = _meminfo) -> Callable[[], int]:
    """AGX Thor's GPU pages a grow takes without reclaim: the freed pages its driver keeps (/proc/meminfo's
    KReclaimable less the slab's SReclaimable) plus torch's cached free blocks; unknown counts as plenty."""

    def pool() -> int:
        m = meminfo()
        if not m or "KReclaimable" not in m:
            return 1 << 62
        return (max(0, m["KReclaimable"] - m.get("SReclaimable", 0)) + int(torch.cuda.memory_reserved())
                - int(torch.cuda.memory_allocated()))

    return pool


__all__ = ["MemoryGate", "NoRoom", "driver_pool", "torch_live"]
