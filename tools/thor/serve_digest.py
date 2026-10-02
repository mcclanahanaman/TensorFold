"""Startup memory digest of a TensorFold Flash Next server log (FR-OPS-17).

    python tools/thor/serve_digest.py LOG [--json OUT]

When the log carries no "-row idle prompt workspace" line, workspace_gib falls back to 0.0 and planned_peak_gib to
startup_estimate_gib. reserve_gib is max(2.0, workspace_gib), the concurrent decoder's memory gate. usable_kv_gib is
kv_room_gib minus reserve_gib, rounded to 2 decimals (None when kv_room_gib is unknown). full_window_streams is
int((kv_room_gib - reserve_gib) // full_window_gib), None when either is unknown or full_window_gib is 0.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_NUM = r"(\d+(?:\.\d+)?)"


def _last_match(pattern: str, text: str):
    found = re.findall(pattern, text)
    if not found:
        return None
    last = found[-1]
    return last if isinstance(last, tuple) else (last,)


def digest(text: str) -> dict:
    start = _last_match(
        r"startup estimate " + _NUM + r" GiB within " + _NUM + r" GiB; native (\d+), "
        r"allocated prompt/reply window (\d+), cache slots (\d+)", text)
    if start:
        startup_estimate_gib, budget_gib = float(start[0]), float(start[1])
        native_window, window, cache_slots = int(start[2]), int(start[3]), int(start[4])
    else:
        startup_estimate_gib = budget_gib = native_window = window = cache_slots = None

    ws = _last_match(
        r"(\d+)-row idle prompt workspace " + _NUM + r" GiB; planned peak " + _NUM + r" GiB", text)
    if ws:
        workspace_rows, workspace_gib, planned_peak_gib = int(ws[0]), float(ws[1]), float(ws[2])
    else:
        workspace_rows, workspace_gib, planned_peak_gib = None, 0.0, startup_estimate_gib

    refused = _last_match(r"TENSORFOLD_PREFILL_ROWS=(\d+): (.*?); idle prompt pieces stay", text)
    refused_rows = int(refused[0]) if refused else None
    refused_why = refused[1] if refused else None

    streams_m = _last_match(r"up to (\d+) streams, each growing to (\d+) prompt/reply tokens", text)
    streams, stream_window = (int(streams_m[0]), int(streams_m[1])) if streams_m else (None, None)

    kv = _last_match(
        r"\(" + _NUM + r" GiB free for their caches, " + _NUM + r" GiB for one at the full window\)", text)
    kv_room_gib, full_window_gib = (float(kv[0]), float(kv[1])) if kv else (None, None)

    passed = _last_match(r"idle prompt pieces (\d+) rows", text)
    pass_rows = int(passed[0]) if passed else None

    copies = _last_match(r"as bf16 copies, " + _NUM + r" GiB", text)
    dense_copies_gib = float(copies[0]) if copies else None

    shadow = _last_match(r"bit cache \(" + _NUM + r" GiB\)", text)
    qsa_shadow_gib = float(shadow[0]) if shadow else None

    loaded = _last_match(r"loaded in " + _NUM + r"s", text)
    loaded_s = float(loaded[0]) if loaded else None

    warmed = _last_match(r"prompt kernels warmed in " + _NUM + r"s", text)
    warmed_s = float(warmed[0]) if warmed else None

    errors = sum(1 for line in text.splitlines() if "Traceback" in line or "out of memory" in line.lower())

    reserve_gib = max(2.0, workspace_gib)
    usable_kv_gib = None if kv_room_gib is None else round(kv_room_gib - reserve_gib, 2)
    if kv_room_gib is None or full_window_gib is None or full_window_gib == 0:
        full_window_streams = None
    else:
        full_window_streams = int((kv_room_gib - reserve_gib) // full_window_gib)

    return {
        "startup_estimate_gib": startup_estimate_gib,
        "budget_gib": budget_gib,
        "native_window": native_window,
        "window": window,
        "cache_slots": cache_slots,
        "workspace_rows": workspace_rows,
        "workspace_gib": workspace_gib,
        "planned_peak_gib": planned_peak_gib,
        "refused_rows": refused_rows,
        "refused_why": refused_why,
        "streams": streams,
        "stream_window": stream_window,
        "kv_room_gib": kv_room_gib,
        "full_window_gib": full_window_gib,
        "pass_rows": pass_rows,
        "dense_copies_gib": dense_copies_gib,
        "qsa_shadow_gib": qsa_shadow_gib,
        "loaded_s": loaded_s,
        "warmed_s": warmed_s,
        "errors": errors,
        "reserve_gib": reserve_gib,
        "usable_kv_gib": usable_kv_gib,
        "full_window_streams": full_window_streams,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Startup memory digest of a TensorFold Flash Next server log.")
    parser.add_argument("log", metavar="LOG", type=Path, help="server log to digest")
    parser.add_argument("--json", metavar="OUT", type=Path, default=None, help="write the digest dict here")
    args = parser.parse_args(argv)
    if not args.log.exists():
        print(f"serve_digest: no such log: {args.log}", file=sys.stderr)
        return 1
    text = args.log.read_text(encoding="utf-8", errors="replace")
    d = digest(text)
    print(f"{args.log.name}: pass rows {d['pass_rows']}, kv room {d['kv_room_gib']} GiB "
          f"(usable {d['usable_kv_gib']}, reserve {d['reserve_gib']}), full-window streams "
          f"{d['full_window_streams']}, window {d['stream_window']}, startup "
          f"{d['startup_estimate_gib']}/{d['budget_gib']} GiB, planned peak {d['planned_peak_gib']} GiB, "
          f"errors {d['errors']}")
    if args.json is not None:
        args.json.write_text(json.dumps(d, indent=1), encoding="utf-8")
    return 0


if __name__ == '__main__':
    sys.exit(main())
