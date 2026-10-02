"""Host memory sampler beside a running TensorFold server on Jetson Thor (FR-OPS-17, ROWS-2): one JSON line a sample.

    python tools/thor/memwatch.py PID OUT.jsonl [--snapshot DIR] [--every 1.0] [--residency-every 1]

Each line: {"t": time.time(), "mem": every /proc/meminfo field in kB, "vm": the VM counters below from /proc/vmstat
(cumulative since boot), "psi": /proc/pressure/memory's "some" and "full" totals in microseconds, "proc": PID's majflt,
minflt, utime and stime (clock ticks), threads, and VmRSS/RssAnon/RssFile in kB, "table": the resident fraction of the
n-gram table's byte ranges in --snapshot's safetensors files (natural_prefill.table_ranges and residency: mincore on
a read-only map), "res_s": the seconds that residency took}. "table" comes every --residency-every samples (else
absent) and is null where mincore fails. Stops when PID exits, or on SIGTERM/SIGINT. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from natural_prefill import residency, table_ranges  # noqa: E402

VM = ("pgmajfault", "pgfault", "pgpgin", "pgpgout", "pswpin", "pswpout", "pgscan_kswapd", "pgscan_direct",
      "pgsteal_kswapd", "pgsteal_direct", "pgsteal_file", "pgscan_file", "allocstall_normal", "allocstall_movable",
      "allocstall_dma", "allocstall_dma32", "workingset_refault_file", "workingset_activate_file",
      "workingset_restore_file", "workingset_refault_anon", "compact_stall", "pgactivate", "pgdeactivate",
      "nr_free_pages", "nr_file_pages", "nr_mapped", "nr_active_file", "nr_inactive_file", "nr_anon_pages")


def meminfo() -> dict:
    out = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts:
            out[key.strip()] = int(parts[0])
    return out


def vmstat() -> dict:
    want, out = set(VM), {}
    for line in Path("/proc/vmstat").read_text().splitlines():
        key, _, value = line.partition(" ")
        if key in want:
            out[key] = int(value)
    return out


def psi() -> dict | None:
    try:
        out = {}
        for line in Path("/proc/pressure/memory").read_text().splitlines():
            kind, *fields = line.split()
            out[kind] = int(dict(f.split("=") for f in fields)["total"])
        return out
    except (OSError, KeyError, ValueError):
        return None


def proc(pid: int) -> dict | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    rest = stat[stat.rindex(")") + 2:].split()      # fields 3.. of stat(5): comm may hold spaces
    out = {"minflt": int(rest[7]), "majflt": int(rest[9]), "utime": int(rest[11]), "stime": int(rest[12]),
           "threads": int(rest[17])}
    for line in status.splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "RssAnon", "RssFile", "RssShmem"):
            out[key] = int(value.split()[0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pid", type=int)
    ap.add_argument("out", type=Path)
    ap.add_argument("--snapshot", type=Path, default=None, help="the checkpoint whose n-gram table residency to log")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between samples")
    ap.add_argument("--residency-every", type=int, default=1, help="table residency every N samples (0: never)")
    args = ap.parse_args()
    ranges = table_ranges(args.snapshot) if args.snapshot else []
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    n = 0
    with args.out.open("a", buffering=1) as f:
        f.write(json.dumps({"t": round(time.time(), 3), "start": {"pid": args.pid, "every": args.every,
                                                                  "table_gib": round(sum(r[2] for r in ranges) / 2**30, 3),
                                                                  "table_ranges": len(ranges)}}) + "\n")
        while not stop:
            t = time.time()
            p = proc(args.pid)
            if p is None:
                break
            rec = {"t": round(t, 3), "mem": meminfo(), "vm": vmstat(), "psi": psi(), "proc": p}
            if ranges and args.residency_every > 0 and n % args.residency_every == 0:
                r0 = time.perf_counter()
                frac = residency(ranges)
                rec["table"] = None if frac is None else round(frac, 5)
                rec["res_s"] = round(time.perf_counter() - r0, 3)
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            n += 1
            time.sleep(max(0.0, args.every - (time.time() - t)))
        f.write(json.dumps({"t": round(time.time(), 3), "end": {"samples": n}}) + "\n")


if __name__ == "__main__":
    main()
