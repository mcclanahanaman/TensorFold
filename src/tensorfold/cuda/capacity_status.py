"""The CUDA server's /v1/capacity: live admission state a client or router can read before sending, so it can back off
before the server would have to refuse (par-1-fix, FR-OPS-17; Brendan 2026-10-05, admission control).

GET reports the state; POST with a request body (a chat completion's or a completion's, as it would be sent) previews
that one request: the prompt tokens it renders to, the kept prefix it would resume now, and whether it would be admitted
now or why not. Read-only, never locked (as health.py's /health): the numbers can be a round old, which is fine for a
capacity check. Contract: docs/api.md's "Capacity and admission" section; keep that doc and this module in step.
"""
from __future__ import annotations

from typing import Any


def _memavailable_mib() -> float | None:
    try:
        for row in open("/proc/meminfo"):
            if row.startswith("MemAvailable:"):
                return round(int(row.split()[1]) / 1024, 1)
    except (OSError, ValueError):     # a capacity read never breaks the server
        pass
    return None


def _pool_mib(gate) -> float | None:
    pool = getattr(gate, "pool", None)
    if pool is None:
        return None
    try:
        return round(pool() / (1 << 20), 1)
    except Exception:          # noqa: BLE001
        return None


def _decoder(app):
    scheduler = getattr(getattr(app, "engine", None), "scheduler", None)
    return scheduler, getattr(scheduler, "decoder", None)


def of(app) -> dict[str, Any]:
    """The app's live capacity now: streams, cache room, kept conversations, evictions and an overloaded flag.

    Byte fields are bytes; ``*_mib`` fields are MiB; token fields are estimates from the decoder's own bytes-per-token
    figure at the full window (``free_cache_tokens_estimate``) or its cache formula (``largest_request_tokens``)."""

    body: dict[str, Any] = {"object": "capacity", "ok": True}
    scheduler, decoder = _decoder(app)
    if decoder is None or scheduler is None:
        body.update(ok=False, reason="no concurrent decoder on this engine (not this family/backend)")
        return body
    gate = getattr(decoder, "memory_gate", None)
    live = len(getattr(decoder, "streams", ()))
    prefilling = len(getattr(decoder, "filling", ()))
    max_streams = int(scheduler.max_streams)
    window = int(getattr(decoder, "capacity", 0) or 0)
    window_bytes = int(getattr(decoder, "window_bytes", 0) or 0)
    bytes_per_token = (window_bytes / window) if window else 0.0
    view = getattr(decoder, "kept_view", None)
    kept = view() if view is not None else [{"tokens": len(ids)} for ids, *_ in list(getattr(decoder, "kept", ()))]
    body.update({
        "live_streams": live, "prefilling_streams": prefilling, "max_streams": max_streams,
        "context_window_tokens": window, "kept_sessions": kept,
        "kept_sessions_total_tokens": sum(k["tokens"] for k in kept),
        "kept_bytes": sum(int(k.get("bytes") or 0) for k in kept),
        "evicted_kept_sessions_total": int(getattr(gate, "evicted_kept", 0) or 0),
        "evicted_kept_by": dict(getattr(decoder, "evicted_by", {}) or {}),
        "waits_total": int(getattr(gate, "waits", 0) or 0), "ended_total": int(getattr(gate, "ends", 0) or 0),
    })
    waiting = getattr(scheduler, "waiting", None)
    refused = dict(getattr(scheduler, "refused", {}) or {})
    queued = (waiting.qsize() if waiting is not None else 0) + (getattr(scheduler, "held", None) is not None)
    body.update({        # requests queued for a seat or room, the bounded wait (None: unbounded) and its refusals
        "queued_requests": queued,
        "admission_wait_s": getattr(scheduler, "max_wait_s", None),
        "refused_total": sum(refused.values()), "refused_by": refused,
    })
    seats = live + prefilling            # a prefilling request holds its seat as a decoding one does
    if gate is not None:
        free_cache_bytes = max(0, gate.room - gate.reserved_total)
        tokens = getattr(decoder, "room_tokens", None)
        largest = tokens() if tokens is not None else window
        body.update({
            "free_cache_bytes": free_cache_bytes,
            "free_cache_tokens_estimate": int(free_cache_bytes / bytes_per_token) if bytes_per_token else None,
            "largest_admissible_prompt_tokens_estimate":
                min(largest, int(free_cache_bytes / bytes_per_token)) if bytes_per_token else largest,
            "largest_request_tokens": largest,
            "reserved_bytes": gate.reserved_total, "held_bytes": gate.held, "room_bytes": gate.room,
            "pooled": bool(getattr(gate, "pooled", False)), "pool_bytes": int(getattr(decoder, "pool_bytes", 0) or 0),
        })
        body["memavailable_mib"] = _memavailable_mib()
        body["pool_headroom_mib"] = _pool_mib(gate)
        body["overloaded"] = seats >= max_streams or free_cache_bytes <= 0
    else:
        body["overloaded"] = seats >= max_streams
    return body


def preview(app, body: dict) -> tuple[int, dict[str, Any]]:
    """(HTTP status, reply) for one request body: its prompt tokens, the kept prefix it would resume, its worst-case
    cache bytes, and ``admissible_now`` with ``reason`` ("length": it can never fit, the window or the cache room;
    "slots": every seat is live; "memory": the room is promised to live requests) and ``retry_after_s``."""

    from tensorfold.server.errors import ContextLengthError, RequestError, error_body

    out: dict[str, Any] = {"object": "capacity.preview", "ok": True}
    scheduler, decoder = _decoder(app)
    if decoder is None or scheduler is None or not hasattr(decoder, "preview"):
        out.update(ok=False, reason="no concurrent decoder on this engine (not this family/backend)")
        return 200, out
    chat = "messages" in body
    if chat:
        from tensorfold.server.prompts import has_images

        if isinstance(body.get("messages"), list) and has_images(body["messages"]):
            out.update(previewed=False, reason=None, note="image requests are not previewed")
            return 200, out
    body = {k: v for k, v in body.items() if k not in ("stream", "stream_options")}
    try:
        prepared = app.prepare(body, chat)
    except ContextLengthError as exc:
        out.update(previewed=True, admissible_now=False, reason="length", message=str(exc))
        return 200, out
    except RequestError as exc:
        return 400, {"error": error_body(exc)}
    got = decoder.preview(list(prepared.prompt), int(prepared.max_tokens), max_streams=int(scheduler.max_streams))
    gate = getattr(decoder, "memory_gate", None)
    out.update(previewed=True, **got, evicted_kept_sessions_total=int(getattr(gate, "evicted_kept", 0) or 0))
    return 200, out


__all__ = ["of", "preview"]
