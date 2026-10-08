"""Requests submit from any thread; one worker thread runs the rounds, and a slow client only fills its own queue."""
# Background requests go last; one decoding yields its lane to a waiting request and re-queues to replay later.

from __future__ import annotations

import heapq
import itertools
import math
import os
import queue
import threading
import time
from typing import Any, Callable

from .memory_gate import MEMORY_RETRY_S, SLOTS_RETRY_S, NoRoom    # noqa: F401  (the retry hints live with NoRoom)
from .streams import Stream


WAIT_ENV = "TENSORFOLD_ADMISSION_WAIT_S"


def admission_wait_s() -> float | None:
    """The longest a request that accepts a refusal waits for a seat or for cache room before its HTTP 503
    (``TENSORFOLD_ADMISSION_WAIT_S``, seconds; 0 refuses at once). Unset or empty: requests wait until admitted."""

    value = os.environ.get(WAIT_ENV, "").strip()
    if not value:
        return None
    try:
        wait = float(value)
    except ValueError:
        wait = -1.0
    if not (math.isfinite(wait) and wait >= 0):
        raise ValueError(f"{WAIT_ENV}: seconds, 0 or more, not {value!r}")
    return wait


class Waiting(queue.PriorityQueue):
    """(stream, box) pairs in arrival order, background streams after every other."""

    def __init__(self) -> None:
        super().__init__()
        self._order = itertools.count()

    def put(self, item, block: bool = True, timeout: float | None = None) -> None:
        super().put((1 if item[0].background else 0, next(self._order), item), block, timeout)

    def get(self, block: bool = True, timeout: float | None = None):
        return super().get(block, timeout)[2]

    def put_many(self, items) -> None:
        """Publish a group before waking the worker, preserving arrival and background priority."""

        with self.not_empty:
            for item in items:
                self._put((1 if item[0].background else 0, next(self._order), item))
                self.unfinished_tasks += 1
            self.not_empty.notify()

    def stop(self) -> None:
        """Wake an idle worker to stop: None comes after every waiting request."""

        super().put((2, next(self._order), None))

    def foreground(self) -> bool:
        """Whether a foreground request waits."""

        with self.mutex:
            return bool(self.queue) and self.queue[0][0] == 0


class Scheduler:
    def __init__(self, decoder: Any, *, max_streams: int = 4, max_wait_s: float | None = None) -> None:
        self.decoder = decoder
        self.max_streams = max_streams
        # a request submitted with ``admitted`` is refused (NoRoom: 503 + Retry-After + reason) once it has waited
        # this long for a seat or for room; None (TENSORFOLD_ADMISSION_WAIT_S unset): every request waits its turn
        self.max_wait_s = admission_wait_s() if max_wait_s is None else float(max_wait_s)
        self.refused = {"slots": 0, "memory": 0}     # requests refused after that wait, by reason (/v1/capacity)
        self.waiting = Waiting()
        self.held: tuple | None = None               # a request waiting for memory, admitted before any other
        self.boxes: dict[int, queue.Queue] = {}
        self.yields = 0                              # background streams that gave up their lane
        if hasattr(decoder, "arrived"):              # a decoder filling prompts lets a new request in between passes
            decoder.arrived = self.waiting.foreground
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def close(self) -> None:
        """Stop the worker once idle and let go of the decoder (its thread held it, and its weights, until now)."""

        self.waiting.stop()
        self.thread.join()
        self.decoder = None

    def submit(self, prompt: list[int], count: int, sampling: Any, draft: bool,
               emit: Callable[[list[int]], bool | None], stop_eos: bool = True, *, vision: Any = None,
               constraint: Any = None, background: bool = False, probabilities: Any = None,
               admitted: Callable[[], None] | None = None) -> dict:
        """Decode one request; ``emit`` runs on the calling thread and returns True to stop. Returns its stats.

        ``admitted`` (on the calling thread, when the request takes its seat, before any token) also lets the request
        be refused while it waits: past ``max_wait_s`` it raises NoRoom ("slots" or "memory", with a retry hint)."""

        box: queue.Queue = queue.Queue()
        stream = Stream(list(prompt), max(1, count), sampling, draft=draft, stop_eos=stop_eos, vision=vision,
                        constraint=constraint, background=background, probabilities=probabilities)
        if admitted is not None and self.max_wait_s is not None:
            stream.queued = time.monotonic()
        cancel = [False]
        stream.emit = lambda new: (box.put(("tokens", new)), cancel[0])[1]
        self.waiting.put((stream, box))
        while True:
            kind, value = box.get()
            if kind == "admitted":
                if admitted is not None:
                    admitted()
            elif kind == "tokens":
                if not cancel[0] and emit(value):
                    cancel[0] = True                 # the client left: the stream ends after its next round
            elif kind == "error":
                raise value
            else:
                return value

    def submit_many(self, requests: list[dict]) -> list[dict]:
        """Queue isolated one-shot requests atomically and return ordered stats, draining errors too."""

        boxes = []
        for r in requests:
            box: queue.Queue = queue.Queue()
            stream = Stream(list(r["prompt"]), max(1, r["count"]), r["sampling"], draft=r.get("draft", True),
                            stop_eos=r.get("stop_eos", True), probabilities=r.get("probabilities"))
            stream.emit = lambda new: False
            boxes.append((stream, box))
        self.waiting.put_many(boxes)
        results, error = [], None
        for _, box in boxes:
            while True:
                kind, value = box.get()
                if kind in ("tokens", "admitted"):
                    continue
                if kind == "error":
                    error = error or value
                    results.append(None)
                else:
                    results.append(value)
                break
        if error is not None:
            raise error
        return results

    def _admit(self, first=None) -> list[Stream]:
        done = []
        while self.decoder.live() < self.max_streams:
            if first is not None:
                (stream, box), first = first, None
            elif self.held is not None:
                (stream, box), self.held = self.held, None
            else:
                try:
                    stream, box = self.waiting.get_nowait()
                except queue.Empty:
                    break
            self.boxes[id(stream)] = box
            try:
                self.decoder.admit(stream)
            except NoRoom as exc:
                self.boxes.pop(id(stream))
                if self.decoder.live():              # waits, first in line, until a live stream finishes
                    self.held = (stream, box)
                    break
                box.put(("error", exc))
                continue
            except Exception as exc:                 # noqa: BLE001  (this request fails, the others go on)
                self.boxes.pop(id(stream)).put(("error", exc))
                continue
            stream.queued = None
            box.put(("admitted", None))
            if stream.done:
                done.append(stream)
        return done

    def _expire(self) -> None:
        """Refuse the requests that accept a refusal and have waited ``max_wait_s`` for admission: NoRoom with the
        reason ("slots": every seat stayed live; "memory": the room stayed promised) and its retry hint. A request
        that has started (a background replay) or that waits patiently (``queued`` None) is never refused."""

        if self.max_wait_s is None:
            return
        now = time.monotonic()
        late = lambda s: s.queued is not None and now - s.queued >= self.max_wait_s   # noqa: E731
        out = []
        if self.held is not None and late(self.held[0]):
            out.append((*self.held, "memory"))          # held: admission found no room for it
            self.held = None
        full = self.decoder.live() >= self.max_streams
        with self.waiting.mutex:
            items = self.waiting.queue
            gone = [e for e in items if e[2] is not None and late(e[2][0])]
            if gone:
                items[:] = [e for e in items if e[2] is None or not late(e[2][0])]
                heapq.heapify(items)
        out += [(*e[2], "slots" if full else "memory") for e in gone]
        for stream, box, reason in out:
            retry = SLOTS_RETRY_S if reason == "slots" else MEMORY_RETRY_S
            self.refused[reason] = self.refused.get(reason, 0) + 1
            what = (f"all {self.max_streams} seats stayed busy" if reason == "slots" else
                    "the cache room stayed promised to the live requests")
            box.put(("error", NoRoom(f"this server is at capacity: {what} for {self.max_wait_s:g} s while this "
                                     f"request waited; retry after {retry:g} s", reason=reason,
                                     retry_after_s=retry)))

    def _yield(self) -> None:
        """Lanes full, a foreground request waiting: the newest background stream (no grammar or images) re-queues."""

        if self.decoder.live() < self.max_streams or not self.waiting.foreground():
            return
        live = list(getattr(self.decoder, "streams", {}).values())
        stream = next((s for s in reversed(live) if s.background and not s.done and s.constraint is None
                       and s.vision is None and len(s.out) < s.count), None)
        if stream is None:
            return
        box = self.boxes.pop(id(stream))
        self.decoder.finish([stream])
        self.yields += 1
        self.waiting.put((stream.continued(), box))

    def _reply(self, s: Stream, kind: str, value: Any) -> None:
        box = self.boxes.pop(id(s), None)            # None: the stream's request has had its reply
        if box is not None:
            box.put((kind, value))

    def _loop(self) -> None:
        while True:
            self._yield()
            idle = not self.decoder.live() and self.held is None
            first = self.waiting.get() if idle else None                              # idle: wait for a request
            if idle and first is None:
                return                                                                # close()
            done = self._admit(first)
            self._expire()                           # after admission: a seat freed this round goes to a waiter first
            try:
                done += self.decoder.round()
            except Exception as exc:                 # noqa: BLE001  (the live requests fail)
                for s in self.decoder.drop():
                    self._reply(s, "error", exc)
            self.decoder.finish(done)
            for s in done:
                self._reply(s, *(("error", s.error) if s.error is not None else ("done", s.stats())))
