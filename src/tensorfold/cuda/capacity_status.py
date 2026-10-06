"""The CUDA server's GET /v1/capacity: live admission state a client or router can read before sending, so it can
back off before the server would have to refuse (par-1-fix, FR-OPS-17; Brendan 2026-10-05, admission control).

Read-only, never locked (as health.py's /health): the numbers can be a round old, which is fine for a capacity
check. Contract: see docs/api.md's "Capacity and admission" section for the field list other code (a router's
seam registry) is meant to build against; keep that doc and this module in step.
"""
from __future__ import annotations

from typing import Any


def _free_mib(torch) -> float | None:
    try:
        free, _ = torch.cuda.mem_get_info()
        return round(free / (1 << 20), 1)
    except Exception:          # noqa: BLE001  a capacity read never breaks the server
        return None


def _pool_mib(gate) -> float | None:
    pool = getattr(gate, "pool", None)
    if pool is None:
        return None
    try:
        return round(pool() / (1 << 20), 1)
    except Exception:          # noqa: BLE001
        return None


def of(app) -> dict[str, Any]:
    """The app's live capacity now: streams, cache room, kept sessions' sizes, and an overloaded flag.

    Every byte field is in MiB (rounded to 0.1) except where the name says "bytes"; every token field is an
    estimate from the decoder's own bytes-per-token-at-full-window figure, which varies a little by how far a
    stream has actually grown, so treat "tokens" fields as a planning estimate, not an exact count.
    """

    body: dict[str, Any] = {"object": "capacity", "ok": True}
    scheduler = getattr(getattr(app, "engine", None), "scheduler", None)
    decoder = getattr(scheduler, "decoder", None)
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
    kept = [{"tokens": len(ids)} for ids, *_ in getattr(decoder, "kept", ())]
    body.update({
        "live_streams": live, "prefilling_streams": prefilling, "max_streams": max_streams,
        "context_window_tokens": window, "kept_sessions": kept, "kept_sessions_total_tokens": sum(k["tokens"] for k in kept),
        "evicted_kept_sessions_total": int(getattr(gate, "evicted_kept", 0) or 0),
        "waits_total": int(getattr(gate, "waits", 0) or 0), "ended_total": int(getattr(gate, "ends", 0) or 0),
    })
    if gate is not None:
        free_cache_bytes = max(0, gate.room - gate.reserved_total)
        body.update({
            "free_cache_bytes": free_cache_bytes,
            "free_cache_tokens_estimate": int(free_cache_bytes / bytes_per_token) if bytes_per_token else None,
            "largest_admissible_prompt_tokens_estimate":
                min(window, int(free_cache_bytes / bytes_per_token)) if bytes_per_token else window,
            "reserved_bytes": gate.reserved_total, "held_bytes": gate.held, "room_bytes": gate.room,
        })
        import torch

        body["memavailable_mib"] = _free_mib(torch)
        body["pool_headroom_mib"] = _pool_mib(gate)
        body["overloaded"] = live >= max_streams or free_cache_bytes <= 0
    else:
        body["overloaded"] = live >= max_streams
    return body


__all__ = ["of"]
