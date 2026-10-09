"""Jetson Thor (sm_110) vision memory: a vision request's host memory budgeted, held, handed back (THOR-VISION-MEM).

Thor's GPU allocates from the host's memory. The Thor KV pool (``thor_kv``) takes all of it at load but the host reserve
(``TENSORFOLD_MEMORY_RESERVE_GIB``) and the on-demand graphs' budget, and the vision workspace
(``TENSORFOLD_VISION_WORKSPACE_MIB``) was carved from the pool's room for the tower's GPU scratch only. A vision
request's host buffers (decoded RGB, the processor's float32 patches and their concatenation, a video's frames)
therefore came out of the host reserve, glibc kept what they freed, and the request's patch arrays stayed alive until
its reply ended:
over the release gauntlet's vision MATCH set MemAvailable fell 3,707 -> 1,025 MiB [vhead-census1] and to 1,392 MiB with
glibc's mmap threshold at 512 KiB [vhead-mmapthr1].

On sm_110 with vision, this path:

1. **Splits the workspace.** The tower's GPU scratch (``gpu_bytes``) stays in the pool, out of the caches' room, as
   before; the rest (``host_bytes``) is left out of the pool (``reserve_pool`` reserves that much less), as host memory
   for vision requests' buffers. The caches' room is what it was: the workspace is the same size, split. The pool
   leaves at least ``HOST_FLOOR_BYTES`` out (``pool_host``), what serving vision was measured to take past the idle;
   that much more comes off the room.
2. **Budgets it.** A request holds its bound (``request_bytes``: decoded pixels, patches, frames, one image's processor
   temporaries) from before its decode until its tower encode ends; another waits (up to the image wait, then 503 as
   before). A request whose bound exceeds the budget runs alone. ``/v1/capacity`` reports the budget.
3. **Hands it back.** Right after the encode the prepared prompt's patch arrays are dropped (its features are on the
   GPU) and ``malloc_trim(0)`` runs on a helper thread, so glibc returns the freed pages at once. A request the server
   may continue past a gate (a thinking budget, a required tool call) encodes its images again, so its arrays and
   hold stay until it ends (``VisionMemory.keep``).

``TENSORFOLD_THOR_VISION_MEM=0`` keeps the earlier path (the whole workspace in the pool, 16 preparation slots, the
arrays kept until the reply ends); any other GPU never takes this path. Nothing here changes what a request computes.
"""

from __future__ import annotations

import ctypes
import os
import threading
import time
import weakref

ENV = "TENSORFOLD_THOR_VISION_MEM"
HOST_ENV = "TENSORFOLD_THOR_VISION_HOST_MIB"     # the host part of the workspace (default: what the GPU part leaves)
MIB = 1 << 20
# the tower's GPU scratch bound: one MAX_PATCHES call's activations peaked at 0.04-0.53 GiB over its start
# [par1fix-mem vision_peak], plus the request's features and the call's bf16 input
GPU_BYTES = 640 * MIB
# the host memory the Thor KV pool leaves free for serving vision (its fit's design point), at least the budget's part:
# over the release's vision MATCH set the server's host memory past its graphs grew up to 551 MiB over its idle (537 on
# the other start) [vmem-a3-c], MemAvailable fell up to 81 MiB more after the App's init, which a refit cannot see (page
# cache, counted about half at Thor's idle MemFree) [vmem-a3-c], and one fit step (64 MiB) covers start-to-start spread
HOST_FLOOR_BYTES = 704 * MIB
TRIM_GAP_S = 0.2                 # a trim at most this often (each vision request asks for one)
KEEP_ATTR = "_thor_vision_keep"  # set on a prepared prompt whose request may encode it again (``VisionMemory.keep``)


def _on(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() not in ("0", "off", "false", "no")


def available(device=None) -> bool:
    """sm_110 (THOR-VISION's own check) and not switched off."""
    from . import thor_vision

    return _on(ENV) and thor_vision.available(device)


def split(workspace: int, gpu_bytes: int) -> tuple[int, int]:
    """(GPU part, host part) of a ``workspace``-byte vision workspace: the GPU part is the tower's scratch bound, capped
    at the workspace; ``TENSORFOLD_THOR_VISION_HOST_MIB`` sets the host part instead (the GPU part is then the rest)."""
    workspace = max(0, int(workspace))
    value = os.environ.get(HOST_ENV, "").strip()
    if value:
        if not value.isdecimal() or int(value) * MIB > workspace:
            raise ValueError(f"{HOST_ENV}: 0 to the vision workspace's {workspace // MIB} MiB, not {value!r}")
        host = int(value) * MIB
        return workspace - host, host
    gpu = min(workspace, max(0, int(gpu_bytes)))
    return gpu, workspace - gpu


def pool_host(host_bytes: int) -> int:
    """The host memory the Thor KV pool leaves out for vision: the budget's ``host_bytes``, or ``HOST_FLOOR_BYTES`` when
    that is more. Only the pool and its design point see it; the budget (what requests hold at once) is unchanged."""
    return max(0, int(host_bytes), HOST_FLOOR_BYTES)


class Hold:
    """``nbytes`` of a ``HostBudget``, released once (``release`` is idempotent and thread-safe)."""

    def __init__(self, budget: "HostBudget", nbytes: int) -> None:
        self.budget, self.nbytes = budget, int(nbytes)
        self._lock = threading.Lock()
        self.released = False

    def release(self) -> None:
        with self._lock:
            if self.released:
                return
            self.released = True
        self.budget._give(self.nbytes)


class HostBudget:
    """Bytes of host memory vision requests may hold at once. ``hold(n, timeout)`` waits until ``n`` fits (``n`` is
    capped at the budget, so a request bigger than it runs alone) and returns a ``Hold``; on timeout it raises
    ``TimeoutError``. First come, first served: a waiter never overtakes an earlier one."""

    def __init__(self, total: int) -> None:
        self.total = max(1, int(total))
        self.used = 0
        self.peak = 0
        self.holds = 0
        self.waits = 0
        self.timeouts = 0
        self._cond = threading.Condition()
        self._queue: list[object] = []

    def hold(self, nbytes: int, timeout: float | None) -> Hold:
        n = max(0, min(int(nbytes), self.total))
        ticket = object()
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._cond:
            self._queue.append(ticket)
            waited = False
            try:
                while not (self._queue[0] is ticket and self.used + n <= self.total):
                    left = None if deadline is None else deadline - time.monotonic()
                    if left is not None and left <= 0:
                        self.timeouts += 1
                        raise TimeoutError(f"vision host budget busy ({self.used / MIB:.0f} of {self.total / MIB:.0f} "
                                           f"MiB held, {n / MIB:.0f} MiB wanted)")
                    waited = True
                    self._cond.wait(left)
            finally:
                self._queue.remove(ticket)
                self._cond.notify_all()
            self.used += n
            self.peak = max(self.peak, self.used)
            self.holds += 1
            self.waits += waited
        return Hold(self, n)

    def _give(self, nbytes: int) -> None:
        with self._cond:
            self.used = max(0, self.used - nbytes)
            self._cond.notify_all()

    def state(self) -> dict:
        with self._cond:
            return {"host_budget_bytes": self.total, "host_held_bytes": self.used, "host_peak_bytes": self.peak,
                    "waiting": len(self._queue), "holds_total": self.holds, "waits_total": self.waits,
                    "timeouts_total": self.timeouts}


def request_bytes(*, images: int, videos: int, image_tokens: int, video_tokens: int, max_total_pixels: int,
                  patch_width: int, frame_bytes: int = 0, streamed: bool = False) -> int:
    """A vision request's host bound before its decode: the decoded RGB of its images (their pixel limit), their bf16
    patches (the visual-token budget, four patches a token), one image's processor temporaries (the processor's float32
    passes over the largest image the budget allows, at most ``image_tokens``), and its videos' frames and their stacked
    copy (``frame_bytes``: the caller's bound for all of them) with their float32 patches. ``streamed`` (one video, no
    image, THOR-VISION-STREAM on) makes one tower call's patches at a time; otherwise every video's patches are made
    whole (four float32 bytes a frame byte) and concatenated into one array, the parts and the copy alive together.

    The bound also covers a gated request's continuation, which encodes again under the request's kept hold: by then
    the decoded pixels, the frames' copies and the processing temporaries are gone, which frees more than that encode's
    host copy of one tower call takes (a streamed video's calls all ran early: its continuation makes none)."""
    total = 0
    if images:
        patches = 4 * int(image_tokens)
        total += 3 * int(max_total_pixels) + 2 * patches * patch_width
        total += 4 * min(patches, 4 * 4096) * patch_width * 3        # one image: float32 patches and two passes
    if videos:
        total += 2 * int(frame_bytes)
        total += 4 * min(4 * int(video_tokens), 16384) * patch_width if streamed else 2 * 4 * int(frame_bytes)
    return int(total)


class Trimmer:
    """``malloc_trim(0)`` on a daemon thread after each ``poke`` (at most every ``TRIM_GAP_S``): glibc hands the pages a
    vision request freed back to the host instead of keeping them in its arenas. A trim never blocks the caller."""

    def __init__(self) -> None:
        try:
            self._trim = ctypes.CDLL("libc.so.6").malloc_trim
            self._trim.argtypes = [ctypes.c_size_t]
        except (OSError, AttributeError):
            self._trim = None
        self._event = threading.Event()
        self.trims = 0
        self.seconds = 0.0
        self._thread = None

    def poke(self) -> None:
        if self._trim is None:
            return
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="tensorfold-vision-trim", daemon=True)
            self._thread.start()
        self._event.set()

    def _run(self) -> None:
        while True:
            self._event.wait()
            self._event.clear()
            t0 = time.perf_counter()
            self._trim(0)
            self.seconds += time.perf_counter() - t0
            self.trims += 1
            time.sleep(TRIM_GAP_S)


class VisionMemory:
    """THOR-VISION-MEM for one tower: the budget, the holds of prepared prompts, and the trimmer."""

    def __init__(self, host_bytes: int, gpu_bytes: int) -> None:
        self.budget = HostBudget(host_bytes)
        self.gpu_bytes = int(gpu_bytes)
        self.trimmer = Trimmer()
        self._holds: dict[int, Hold] = {}
        self._lock = threading.Lock()
        self.released_bytes = 0

    def hold(self, nbytes: int, timeout: float | None) -> Hold:
        return self.budget.hold(nbytes, timeout)

    def attach(self, prepared, hold: Hold) -> None:
        """``hold`` now belongs to ``prepared``: released after its encode (``done``), or when it is collected."""
        with self._lock:
            self._holds[id(prepared)] = hold
        weakref.finalize(prepared, self._drop, id(prepared), hold)

    def _drop(self, key: int, hold: Hold) -> None:
        with self._lock:
            if self._holds.get(key) is hold:
                del self._holds[key]
        hold.release()
        self.trimmer.poke()

    @staticmethod
    def keep(prepared) -> None:
        """``prepared`` may be encoded again (the server's gated continuation re-encodes its images): its arrays and
        hold stay until the request drops it (``attach``'s finalizer), not just past its first encode. The
        continuation's encode runs under that hold (``request_bytes`` covers it); a second hold from the same budget
        could wait forever on this one."""
        object.__setattr__(prepared, KEEP_ATTR, True)

    def done(self, prepared) -> None:
        """After ``prepared``'s encode (or its failure): its patch arrays dropped, its hold released, a trim asked;
        a kept prompt (``keep``) only asks the trim."""
        if getattr(prepared, KEEP_ATTR, False):
            self.trimmer.poke()
            return
        self.released_bytes += release_arrays(prepared)
        with self._lock:
            hold = self._holds.pop(id(prepared), None)
        if hold is not None:
            hold.release()
        self.trimmer.poke()

    def state(self) -> dict:
        out = self.budget.state()
        out.update(gpu_workspace_bytes=self.gpu_bytes, released_bytes_total=self.released_bytes,
                   trims_total=self.trimmer.trims, trim_seconds_total=round(self.trimmer.seconds, 3))
        return out


def release_arrays(prepared) -> int:
    """Drop a prepared prompt's patch arrays and early video features (its encode has run: nothing reads them again);
    zero-row arrays of the same width keep its shape checks meaningful. Returns the bytes the arrays held."""
    import numpy as np

    freed = 0
    for name in ("pixel_values", "video_pixel_values"):
        value = getattr(prepared, name, None)
        if value is None:
            continue
        freed += int(getattr(value, "nbytes", 0) or 0)
        width = int(value.shape[1]) if len(getattr(value, "shape", ())) == 2 else 0
        object.__setattr__(prepared, name, np.empty((0, width), dtype=np.float32))
    if getattr(prepared, "video_features", None) is not None:
        object.__setattr__(prepared, "video_features", None)
    return freed


__all__ = ["ENV", "HOST_ENV", "HOST_FLOOR_BYTES", "HostBudget", "Hold", "Trimmer", "VisionMemory", "available",
           "pool_host", "release_arrays", "request_bytes", "split"]
