"""Stream caches take memory as they grow: the room the startup plan left them, what they hold, and a reserve."""

from __future__ import annotations

from typing import Callable

from tensorfold.server.errors import CapacityError

from .capacity import cuda_limit_bytes


class NoRoom(CapacityError):
    """A request that can't start until a live stream finishes and frees its caches: HTTP 503, retry shortly.

    par-1-fix (Brendan 2026-10-05, admission control): was ``RuntimeError``. Every existing catch site names
    ``NoRoom`` explicitly (unaffected); the change is so the many ``isinstance(exc, CapacityError)`` checks already
    in the HTTP layer (cuda/http.py, server/http.py, server/anthropic.py) map it to 503 with no new code there.
    ``reason`` is a short machine-readable tag (``"memory"``, ``"slots"`` or ``"length"``) a client or router can
    key on without parsing the message; ``retry_after_s`` is this gate's own best estimate of a useful wait, or
    None where no better guess exists than "the usual poll interval" (the caller picks a default)."""

    def __init__(self, message: str, *, reason: str = "memory", retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.reason = reason
        self.retry_after_s = retry_after_s


class MemoryGate:
    """``room`` bytes for every stream's growing caches; ``fits`` also asks the host (MemAvailable on a unified GPU)."""

    def __init__(self, room: int, reserve: int, live: Callable[[], int] | None = None,
                 pool: Callable[[], int] | None = None) -> None:
        self.room, self.reserve, self.held = int(room), int(reserve), 0
        self.live = live
        self.pool = pool                 # AGX Thor's wide prompt pieces: the GPU pages a grow takes without reclaim
        self.waits = self.ends = 0
        self.evicted_kept = 0             # par-1-fix: LRU-evicted kept prefixes, counted so /v1/capacity can report it
        # par-1-fix: admission reserves a live stream's full worst-case extent (prompt + its declared max tokens,
        # clamped to the window) for its whole lifetime, not just what it currently holds. Simpler and safe
        # (never under-reserves) at the cost of some utilisation: a stream that ends up needing less than its
        # worst case still occupies that much of the ledger until it finishes. This is what lets admission refuse
        # a new request BEFORE it starts, rather than an already-running one being squeezed later (multi.py
        # _make_room's kill-the-newest fallback is then unreachable in normal operation: see its docstring).
        # Ledgered with promise()/unpromise() below (not named reserve()/unreserve(): self.reserve, right above,
        # is already the margin-bytes int every grow() checks against; the names must not collide with it).
        self.reserved: dict[int, int] = {}
        # Thor KV pool (families/qwen4_exp/cuda/thor_kv.py): ``room`` is a block reserved in torch's allocator at load;
        # what lives outside it (the on-demand decode graphs) asks ``outside`` (the host's memory past the reserve).
        self.pooled = False
        self.system: Callable[[], int] | None = None

    @property
    def reserved_total(self) -> int:
        return sum(self.reserved.values())

    def admits(self, extra: int) -> bool:
        """Whether reserving ``extra`` more bytes (a new stream's full worst-case extent) still fits ``room``."""

        return self.reserved_total + int(extra) <= self.room

    def promise(self, key: int, extra: int) -> None:
        """Ledger ``extra`` bytes against ``key`` (named ``promise``: ``self.reserve`` is already the margin int)."""

        self.reserved[key] = int(extra)

    def unpromise(self, key: int) -> int:
        return self.reserved.pop(key, 0)

    def fits(self, extra: int) -> bool:
        """Whether ``extra`` more bytes can be allocated now (a resize holds its old buffers until the copy lands)."""

        if self.held + int(extra) > self.room - self.reserve:
            return False
        return self.live is None or self.live() >= int(extra) + self.reserve

    def take(self, extra: int) -> None:
        self.held += int(extra)

    def give(self, freed: int) -> None:
        self.held = max(0, self.held - int(freed))

    def outside(self, extra: int) -> bool:
        """Whether ``extra`` bytes allocated outside the pool (``pooled``) leave the host its reserve; True without one."""

        return self.system is None or self.system() >= int(extra)

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
