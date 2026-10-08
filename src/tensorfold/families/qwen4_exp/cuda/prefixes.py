"""Select and retain prompt prefixes without consuming a longer chain when a spare slot can hold a copy.

A slot's kept chain is one conversation's resume points. When slots or memory run short, the least recently used idle
slot goes first (``owner.used``: when each slot last started or finished a request), so a conversation that keeps
coming back keeps its prefix while the one left idle longest runs cold; a slot's inner points (shorter prefixes inside
its own chain) go before any conversation's longest point. Every removal of a whole chain is counted
(``owner._evict_kept``, by cause) for ``/v1/capacity``, and so is every point dropped while its chain stays
(``owner.points_dropped``: an inner point past ``keep``, or, when no idle slot can go, the oldest point, which may be a
live conversation's own resume point).
"""


def _best(kept, prompt, busy=()):
    return max((k for k in kept if id(k[1]) not in busy and len(k[0]) < len(prompt)
                and prompt[:len(k[0])] == k[0]), key=lambda k: len(k[0]), default=None)


def _longer(kept, entry):
    return any(k[1] is entry[1] and len(k[0]) > len(entry[0]) for k in kept)


def _spare(owner, protect):
    """A free slot, else the least recently used idle kept slot (never ``protect``), freed and counted; None if none."""

    if owner.free:
        return owner.free.pop()
    if owner._evict_kept(None, protect=protect, cause="slots"):
        return owner.free.pop()
    return None


def slot_for(owner, prompt: list[int], reuse: bool):
    """Copy a fork into a spare slot (a free one, or the least recently used idle one); else extend a slot's own chain.

    A prompt matching only a shorter point of a slot whose chain goes on (another conversation on the same system
    prompt) copies that point into a spare slot rather than taking the slot: its longer chain stays kept."""

    busy = owner._busy()
    best = _best(owner.kept, prompt) if reuse else None
    fork = best is not None and (id(best[1]) in busy or _longer(owner.kept, best))
    if fork:
        spare = _spare(owner, best[1])
        if spare is not None:
            try:
                if owner._grow(spare, len(prompt) + owner.depth + 2, protect=best[1]):
                    spare.copy_prefix(best[1], len(best[0]), best[2]["mtp_len"])
                    return spare, {"state": best[2], "tail": best[3]}, len(best[0])
            except Exception:
                owner.free.append(spare)
                owner._shrink(spare, force=True)
                raise
            owner.free.append(spare)
        best = _best(owner.kept, prompt, busy) if reuse else None
        if best is not None and owner.free and _longer(owner.kept, best):
            best = None
    if best is not None:
        n = len(best[0])
        owner.kept = [k for k in owner.kept if k[1] is not best[1]
                      or len(k[0]) <= n and best[0][:len(k[0])] == k[0]]
        return best[1], {"state": best[2], "tail": best[3]}, n
    spare = _spare(owner, None)
    if spare is None:
        raise RuntimeError("no free stream slot")
    return spare, None, 0


def remember(owner, ids, st, snap, tail) -> None:
    """Keep each slot's prefix chain; past ``owner.keep`` points, inner points go first, then idle slots' whole chains,
    least recently used first (``owner._evict_kept``: their slots shrink and join the free list)."""

    gone = [k[1] for k in owner.kept if k[0] == ids]
    owner.kept = [k for k in owner.kept if k[0] != ids] + [(ids, st, snap, tail)]
    used = getattr(owner, "used", {})

    def lru(k):
        return used.get(id(k[1]), 0.0), len(k[0])

    while len(owner.kept) > owner.keep:
        top: dict[int, int] = {}
        for k in owner.kept:
            top[id(k[1])] = max(top.get(id(k[1]), 0), len(k[0]))
        inner = [k for k in owner.kept if len(k[0]) < top[id(k[1])]]
        if inner:
            drop = min(inner, key=lru)
            owner.kept = [k for k in owner.kept if k is not drop]
            owner.points_dropped = getattr(owner, "points_dropped", 0) + 1
            continue
        if owner._evict_kept(st, protect=None, cause="entries"):
            continue
        gone.append(owner.kept.pop(0)[1])            # no idle slot can go: the oldest point, counted as a point
        owner.points_dropped = getattr(owner, "points_dropped", 0) + 1
    busy = owner._busy()
    for old in gone:
        if old is not st and id(old) not in busy and all(k[1] is not old for k in owner.kept) and \
                all(f is not old for f in owner.free):
            owner.free.append(old)
