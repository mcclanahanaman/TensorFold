"""Flash Next's n-gram table on Jetson Thor: why it is not locked, what a cold table costs a prompt, and the options.

    python tools/thor/ngram_residency.py MODEL_DIR OUT_DIR [--tokens 32768] [--parallel 4] [--kv-dtype int8]
        [--pressure 131072] | [--held GIB]

1. Limits: RLIMIT_MEMLOCK, capabilities (CAP_IPC_LOCK), cgroup memory, mlock and mlock2 on a slice of the table.
2. The engine as ``tensorfold serve MODEL --backend cuda --kv-dtype int8 --parallel 4`` builds it: the startup budget's
   room against the table's size (the lock decision) and what the KV gate counts as free.
3. One natural-text and one synthetic prompt, warm (the table as read at startup) and cold (its pages dropped:
   MADV_DONTNEED on the engine's maps, then POSIX_FADV_DONTNEED), each option run on the same prompt (kept prompt ends
   are forgotten between runs, so every run fills from row 0; the head logits' hash must not change):

     default       the engine as it is
     threads64     the gather's pool at 64 threads (16 today)
     madv_normal   the table's maps with kernel read-ahead (MADV_NORMAL; the engine sets MADV_RANDOM)
     rows_ahead    at admission, POSIX_FADV_WILLNEED on every page the prompt's rows touch, pass order, on a thread
     stage_ahead   a pass's gather before ``stage``'s wait on the previous pass (prototype, a run-time patch)
     preread       the table read once more (``table.prefetch()``), then the prompt
     willneed      MADV_WILLNEED on the whole table, the prompt once it is resident again
     mlock         the table locked (mlock), then the prompt; the drop attempted again, then the prompt again

``--held GIB`` instead holds device memory until MemAvailable is about GIB (a server whose caches have grown) and runs
each option with the table cold under that pressure, where the gathers' page faults reclaim as they go.

Nothing in the engine's source changes: the options are run-time patches in this process. Writes residency.json (or
residency_held.json) and ledger_metrics.json (prefill_32k: the synthetic prompt, warm, default path).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import prefill_profile as pp  # noqa: E402

PAGE = mmap.PAGESIZE
LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.mlock.argtypes = LIBC.munlock.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
LIBC.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
MADV_NORMAL, MADV_RANDOM, MADV_WILLNEED, MADV_DONTNEED = 0, 1, 3, 4
CAP_IPC_LOCK = 14


def meminfo() -> dict:
    out = {}
    for line in open("/proc/meminfo"):
        k, v, *_ = line.split()
        out[k.rstrip(":")] = int(v) * 1024
    return out


def gib(v: float) -> float:
    return round(v / 2**30, 3)


def read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def limits() -> dict:
    """The memory-lock limit, capabilities, cgroup memory files and the table's mount."""

    out: dict = {"uname": os.uname().release}
    for line in (read("/proc/self/limits") or "").splitlines():
        if "locked memory" in line.lower():
            out["rlimit_memlock"] = line.split()[3:5]
    caps = {}
    for line in (read("/proc/self/status") or "").splitlines():
        if line.startswith(("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")):
            k, v = line.split(":")
            caps[k] = v.strip()
    out["caps"] = caps
    out["cap_ipc_lock_effective"] = bool(int(caps.get("CapEff", "0"), 16) >> CAP_IPC_LOCK & 1)
    cg = {}
    for name in ("memory.max", "memory.high", "memory.current", "memory.swap.max", "memory.min", "memory.low"):
        v = read(f"/sys/fs/cgroup/{name}")
        if v is not None:
            cg[name] = v
    for name in ("memory.limit_in_bytes", "memory.memsw.limit_in_bytes", "memory.usage_in_bytes"):
        v = read(f"/sys/fs/cgroup/memory/{name}")
        if v is not None:
            cg["v1:" + name] = v
    stat = read("/sys/fs/cgroup/memory.stat") or ""
    cg["stat"] = {k: v for k, v in (line.split() for line in stat.splitlines() if line)
                  if k in ("file", "anon", "unevictable", "file_mapped", "active_file", "inactive_file")}
    out["cgroup"] = cg
    out["vm"] = {k: read(f"/proc/sys/vm/{k}") for k in ("swappiness", "vfs_cache_pressure", "min_free_kbytes",
                                                         "overcommit_memory", "watermark_scale_factor")}
    out["mounts"] = [line for line in (read("/proc/mounts") or "").splitlines() if " /hf " in line]
    return out


def try_mlock(arr: np.ndarray, nbytes: int) -> dict:
    """mlock, then mlock2(MLOCK_ONFAULT), on the first ``nbytes`` of a table array: return codes and errno."""

    at = arr.ctypes.data
    start = at - at % PAGE
    size = nbytes + (at - start)
    out = {}
    t = time.perf_counter()
    rc = LIBC.mlock(ctypes.c_void_p(start), ctypes.c_size_t(size))
    out["mlock"] = {"rc": rc, "errno": ctypes.get_errno() if rc else 0, "s": round(time.perf_counter() - t, 3)}
    if rc == 0:
        LIBC.munlock(ctypes.c_void_p(start), ctypes.c_size_t(size))
    if hasattr(LIBC, "mlock2"):
        LIBC.mlock2.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        rc = LIBC.mlock2(ctypes.c_void_p(start), ctypes.c_size_t(size), 1)          # MLOCK_ONFAULT
        out["mlock2_onfault"] = {"rc": rc, "errno": ctypes.get_errno() if rc else 0}
        if rc == 0:
            LIBC.munlock(ctypes.c_void_p(start), ctypes.c_size_t(size))
    out["bytes"] = size
    return out


class Table:
    """The engine's NVFP4 n-gram table: its arrays (codes, block scales) and their files."""

    def __init__(self, w) -> None:
        layer = next(layer for layer in w.layers if layer.ple is not None)
        self.layer, self.t, self.ngram = layer, layer.ple.table, layer.ple.ngram
        self.arrays = list(getattr(self.t, "values", [])) + list(getattr(self.t, "scales", []))
        self.fds: dict[str, int] = {}

    def spans(self):
        for arr in self.arrays:
            at = arr.ctypes.data
            start = at - at % PAGE
            yield arr, start, arr.nbytes + (at - start)

    def resident(self) -> float:
        have, total = pp.resident(self.arrays)
        return have / max(1, total)

    def advise(self, advice: int) -> list[int]:
        return [LIBC.madvise(ctypes.c_void_p(s), ctypes.c_size_t(n), advice) for _, s, n in self.spans()]

    def fd(self, arr) -> int:
        name = str(arr.filename)
        if name not in self.fds:
            self.fds[name] = os.open(name, os.O_RDONLY)
        return self.fds[name]

    def drop(self) -> dict:
        """Every page out of the page cache: unmapped from the engine's maps, then dropped from the file cache."""

        t = time.perf_counter()
        rcs = self.advise(MADV_DONTNEED)
        for arr in self.arrays:
            os.posix_fadvise(self.fd(arr), int(arr.offset), int(arr.nbytes), os.POSIX_FADV_DONTNEED)
        return {"s": round(time.perf_counter() - t, 2), "madvise_rc": sorted(set(rcs)), "resident": self.resident()}

    def lock(self) -> dict:
        t = time.perf_counter()
        before = meminfo()
        rcs = []
        for _, s, n in self.spans():
            rc = LIBC.mlock(ctypes.c_void_p(s), ctypes.c_size_t(n))
            rcs.append((rc, ctypes.get_errno() if rc else 0))
        after = meminfo()
        return {"s": round(time.perf_counter() - t, 1), "rc_errno": sorted(set(rcs)),
                "MemAvailable_gib": [gib(before["MemAvailable"]), gib(after["MemAvailable"])],
                "Mlocked_gib": [gib(before.get("Mlocked", 0)), gib(after.get("Mlocked", 0))],
                "Unevictable_gib": [gib(before.get("Unevictable", 0)), gib(after.get("Unevictable", 0))]}

    def unlock(self) -> None:
        for _, s, n in self.spans():
            LIBC.munlock(ctypes.c_void_p(s), ctypes.c_size_t(n))

    def pages(self, ids: np.ndarray) -> list[tuple[int, int, int]]:
        """(fd, first byte, length) runs of the pages rows ``ids`` occupy, codes and scales, in order of first use."""

        shard, local = self.t._where(ids)
        runs = []
        for arrays in (self.t.values, self.t.scales):
            rowb = arrays[0].shape[1] * arrays[0].dtype.itemsize
            keyed = {}
            for f in np.unique(shard):
                arr = arrays[f]
                off = int(arr.offset) + local[shard == f] * rowb
                pages = np.unique(np.concatenate([off // PAGE, (off + rowb - 1) // PAGE]))
                keyed[f] = (self.fd(arr), pages)
            for fd, pages in keyed.values():
                breaks = np.nonzero(np.diff(pages) != 1)[0] + 1
                for run in np.split(pages, breaks):
                    runs.append((fd, int(run[0]) * PAGE, int(len(run)) * PAGE))
        return runs


# ------------------------------------------------------------------------------------------------ options

STATE: dict = {"stage_ahead": False, "rows_ahead": False, "threads": None, "ahead_log": []}


def install(engine, table: Table) -> None:
    """The run-time options' hooks (each off until STATE turns it on)."""

    from tensorfold.families.qwen4_exp.cuda import forward, multi

    original_stage = multi.stage

    def stage(w, b, windows):
        if not STATE["stage_ahead"] or w.x3 is not None or \
                any(getattr(layer.ple.table, "bits", 4) != 16 for layer in w.layers if layer.ple is not None):
            return original_stage(w, b, windows)
        return stage_ahead(w, b, windows)

    multi.stage = stage
    admit = multi.MultiDecoder.admit

    def admitted(dec, s):
        if STATE["rows_ahead"]:
            rows_ahead(table, list(s.prompt), dec.prefill_rows)
        return admit(dec, s)

    multi.MultiDecoder.admit = admitted
    del forward


def stage_ahead(w, b, windows):
    """``forward.stage`` with each n-gram gather before the wait on the pinned buffers (the previous pass's copies):
    the gather's page faults and copies run while the GPU still works on that pass. The same rows go to the same
    buffers; only the host's order changes (prototype)."""

    segs = []
    for st, tokens in windows:
        a0 = segs[-1][2] if segs else 0
        if st.pos + len(tokens) > st.capacity:
            raise ValueError("context past the cache capacity")
        segs.append((st, a0, a0 + len(tokens)))
    R = segs[-1][2]
    if R > b.rows:
        raise ValueError(f"window of {R} rows, buffers hold {b.rows}")
    gathered = []
    for layer in w.layers:
        if layer.ple is not None:
            p = layer.ple
            for (st, tokens), (_, a0, _) in zip(windows, segs):
                toks = np.asarray(tokens, dtype=np.int64)
                ids = p.ngram.ids(st.ple_history, toks)
                st.ple_last = (st.ple_history, toks)
                gathered.append((a0 * (ids.size // len(toks)), p.table.gather(ids)))
    b.staged.synchronize()               # the previous step's copies out of the pinned buffers are done
    b.ids_host[:R].numpy()[:] = np.asarray([t for _, tokens in windows for t in tokens], dtype=np.int32)
    b.ids[:R].copy_(b.ids_host[:R], non_blocking=True)
    for at, values in gathered:
        rows = slice(at, at + values.shape[0])
        b.ple_hv[rows].view(torch.int16).numpy()[:] = values.view(np.int16)
        b.ple_v[rows].copy_(b.ple_hv[rows], non_blocking=True)
    b.staged.record()
    return segs


def rows_ahead(table: Table, prompt: list[int], rows: int) -> None:
    """POSIX_FADV_WILLNEED on the pages of the prompt's rows, a pass at a time in order, on a thread."""

    toks = np.asarray(prompt, dtype=np.int64)
    ids = table.ngram.ids(table.ngram.initial_history(), toks)
    log = {"t0": time.perf_counter(), "passes": []}
    STATE["ahead_log"].append(log)

    def work() -> None:
        for a in range(0, len(toks), rows):
            t = time.perf_counter()
            runs = table.pages(ids[a:a + rows])
            for fd, at, n in runs:
                os.posix_fadvise(fd, at, n, os.POSIX_FADV_WILLNEED)
            log["passes"].append((round(time.perf_counter() - log["t0"], 3), len(runs),
                                  round(time.perf_counter() - t, 3)))

    threading.Thread(target=work, daemon=True, name="ngram-rows-ahead").start()


def set_threads(table: Table, n: int | None) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from tensorfold.families.qwen4_exp import host_table

    if n is None:
        n = 16
    host_table.GATHER_THREADS = n
    table.t._pool = ThreadPoolExecutor(n, thread_name_prefix="ngram-gather")


# ------------------------------------------------------------------------------------------------ main

def forget(engine) -> None:
    """Free every kept prompt end, so the next prompt (the same ids too) fills from row 0."""

    multi = engine.multi
    while multi._evict_kept(None):
        pass


def run(engine, ids, label: str, table: Table, out: list, extra: dict | None = None) -> dict:
    forget(engine)
    before = table.resident()
    got = pp.run_prompt(engine, ids)
    row = {"label": label, "tokens": got["tokens"], "s": round(got["seconds"], 3), "tok_s": round(got["tok_s"], 1),
           "first": got["first"], "logits": got["logits_sha"], "resident_before": round(before, 4),
           "resident_after": round(table.resident(), 4), "MemAvailable_gib": gib(meminfo()["MemAvailable"]),
           **(extra or {})}
    out.append(row)
    print(f"[ngram] {label}: {row['s']:.2f} s = {row['tok_s']:.0f} tok/s; resident {100 * before:.1f}% -> "
          f"{100 * row['resident_after']:.1f}%; logits {row['logits']}; first {row['first']}"
          + (f"; {extra}" if extra else ""), flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--kv-dtype", default="int8")
    ap.add_argument("--pressure", type=int, default=131072, help="a synthetic prompt of this many tokens, then cold?")
    ap.add_argument("--held", type=float, default=0.0,
                    help="instead: hold device memory until MemAvailable is this many GiB, then each option cold")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.held:
        return held(args)
    report: dict = {"limits": limits(), "runs": []}
    print(f"[ngram] limits: {json.dumps(report['limits'])}", flush=True)
    t0 = time.perf_counter()
    engine = pp.build_engine(args.model, args.parallel, args.kv_dtype)
    plan = engine.capacity_plan
    table = Table(engine.w)
    m = meminfo()
    from tensorfold.cuda.capacity import available_bytes

    room = plan["budget_bytes"] - plan["total_bytes_estimate"]
    report["engine"] = {
        "load_s": round(time.perf_counter() - t0, 1), "budget_gib": gib(plan["budget_bytes"]),
        "estimate_gib": gib(plan["total_bytes_estimate"]), "room_gib": gib(room),
        "table_gib": gib(table.t.nbytes), "table_class": type(table.t).__name__,
        "lock_called": room >= table.t.nbytes, "kv_gate_room_gib": gib(engine.multi.memory_gate.room),
        "available_now_gib": gib(available_bytes(torch)), "MemAvailable_gib": gib(m["MemAvailable"]),
        "Cached_gib": gib(m["Cached"]), "Mlocked_gib": gib(m.get("Mlocked", 0)), "resident": table.resident()}
    print(f"[ngram] engine: {json.dumps(report['engine'])}", flush=True)
    report["mlock_slice"] = try_mlock(table.arrays[0], min(1 << 30, table.arrays[0].nbytes))
    print(f"[ngram] mlock on 1 GiB of the table: {report['mlock_slice']}", flush=True)
    install(engine, table)
    runs = report["runs"]
    syn = pp.prompt_ids(args.model, args.tokens, 424242)
    nat = pp.prompt_ids(args.model, args.tokens, 515151, natural=True)
    # warm (as read at startup)
    run(engine, syn, "warm default synthetic", table, runs)
    run(engine, nat, "warm default natural", table, runs)
    STATE["stage_ahead"] = True
    run(engine, syn, "warm stage_ahead synthetic", table, runs)
    run(engine, nat, "warm stage_ahead natural", table, runs)
    STATE["stage_ahead"], STATE["rows_ahead"] = False, True
    run(engine, syn, "warm rows_ahead synthetic", table, runs)
    STATE["stage_ahead"] = True
    run(engine, syn, "warm stage_ahead+rows_ahead synthetic", table, runs)
    STATE["stage_ahead"] = STATE["rows_ahead"] = False

    def cold(label: str, **kw) -> dict:
        dropped = table.drop()
        return run(engine, nat, label, table, runs, {"drop": dropped, **kw})

    cold("cold default natural")
    set_threads(table, 64)
    cold("cold threads64 natural")
    set_threads(table, None)
    table.advise(MADV_NORMAL)
    cold("cold madv_normal natural")
    table.advise(MADV_RANDOM)
    STATE["rows_ahead"] = True
    cold("cold rows_ahead natural")
    STATE["rows_ahead"], STATE["stage_ahead"] = False, True
    cold("cold stage_ahead natural")
    STATE["rows_ahead"] = True
    cold("cold stage_ahead+rows_ahead natural")
    set_threads(table, 64)
    cold("cold stage_ahead+rows_ahead+threads64 natural")
    set_threads(table, None)
    STATE["rows_ahead"] = STATE["stage_ahead"] = False
    report["ahead_log"] = [{"passes": log["passes"]} for log in STATE["ahead_log"]]
    # whole-table warm-ups from cold
    dropped = table.drop()
    t = time.perf_counter()
    table.t.prefetch()
    pre = {"s": round(time.perf_counter() - t, 1), "resident": table.resident(), "drop": dropped,
           "gb_s": round(table.t.nbytes / (time.perf_counter() - t) / 1e9, 2)}
    print(f"[ngram] pre-read from cold: {pre}", flush=True)
    run(engine, nat, "preread natural", table, runs, {"preread": pre})
    dropped = table.drop()
    t = time.perf_counter()
    table.advise(MADV_WILLNEED)
    issued = time.perf_counter() - t
    while table.resident() < 0.95 and time.perf_counter() - t < 300:
        time.sleep(1.0)
    will = {"issue_s": round(issued, 2), "s": round(time.perf_counter() - t, 1), "resident": table.resident(),
            "drop": dropped}
    table.advise(MADV_RANDOM)
    print(f"[ngram] MADV_WILLNEED from cold: {will}", flush=True)
    run(engine, nat, "willneed natural", table, runs, {"willneed": will})
    # mlock: from cold, its cost to the KV gate's view, a drop attempt while locked
    dropped = table.drop()
    gate_before = gib(available_bytes(torch))
    locked = table.lock()
    locked["available_gib"] = [gate_before, gib(available_bytes(torch))]
    locked["resident"] = table.resident()
    locked["drop_from"] = dropped
    print(f"[ngram] mlock from cold: {locked}", flush=True)
    run(engine, nat, "mlock natural", table, runs, {"mlock": locked})
    again = table.drop()
    run(engine, nat, "mlock dropped natural", table, runs, {"drop": again})
    table.unlock()
    report["mlock"] = locked
    # real pressure, from a warm table: a long synthetic prompt (its caches grow and stay kept), then an allocation
    # the size of more streams' caches; the table's residency after each, then the natural prompt as served
    table.t.prefetch()
    pressure: dict = {"resident_start": table.resident(), "MemAvailable_start_gib": gib(meminfo()["MemAvailable"])}
    if args.pressure:
        big = pp.prompt_ids(args.model, args.pressure, 616161)
        got = pp.run_prompt(engine, big)             # kept: its slot holds its caches
        pressure.update({"long_tok_s": round(got["tok_s"], 1), "resident_after_long": table.resident(),
                         "MemAvailable_after_long_gib": gib(meminfo()["MemAvailable"])})
        print(f"[ngram] pressure: synthetic {args.pressure}: {got['tok_s']:.0f} tok/s, table resident "
              f"{100 * pressure['resident_after_long']:.1f}%", flush=True)
    for size in (8, 16):
        block = torch.empty((size << 30,), dtype=torch.uint8, device="cuda")
        block.fill_(1)
        torch.cuda.synchronize()
        pressure[f"resident_after_{size}gib"] = table.resident()
        pressure[f"MemAvailable_after_{size}gib"] = gib(meminfo()["MemAvailable"])
        print(f"[ngram] pressure: +{size} GiB allocated: table resident "
              f"{100 * pressure[f'resident_after_{size}gib']:.1f}%, MemAvailable "
              f"{pressure[f'MemAvailable_after_{size}gib']} GiB", flush=True)
        del block
        torch.cuda.empty_cache()
    report["pressure"] = pressure
    run(engine, nat, "after pressure default natural", table, runs)
    forget(engine)
    by = {r["label"]: r for r in runs}
    hashes = {}
    for r in runs:
        kind = r["label"].split()[-1]
        if kind in ("synthetic", "natural"):
            hashes.setdefault(kind, set()).add(r["logits"])
    report["identical_logits"] = {k: len(v) == 1 for k, v in hashes.items()}
    passed = sum(1 for v in report["identical_logits"].values() if v)
    (args.out / "residency.json").write_text(json.dumps(report, indent=1, default=str))
    warm = by.get("warm default synthetic")
    (args.out / "ledger_metrics.json").write_text(json.dumps({
        "prefill_32k": warm["tok_s"] if warm else None, "kv_gib": report["engine"]["kv_gate_room_gib"],
        "tests_passed": passed, "tests_failed": len(report["identical_logits"]) - passed}))
    print(f"[ngram] identical head logits per prompt across options: {report['identical_logits']}", flush=True)
    print(f"[ngram] done in {time.perf_counter() - t0:.0f}s", flush=True)


def held(args) -> None:
    """Memory held so MemAvailable stays near ``--held`` GiB (as a server whose caches have grown), the table cold
    before each option: the gathers' page faults now reclaim as they go."""

    t0 = time.perf_counter()
    engine = pp.build_engine(args.model, args.parallel, args.kv_dtype)
    table = Table(engine.w)
    install(engine, table)
    nat = pp.prompt_ids(args.model, args.tokens, 515151, natural=True)
    syn = pp.prompt_ids(args.model, args.tokens, 424242)
    report: dict = {"held_gib": args.held, "runs": []}
    runs = report["runs"]
    run(engine, syn, "unheld warm default synthetic", table, runs)
    run(engine, nat, "unheld warm default natural", table, runs)
    blocks: list = []

    def hold() -> dict:
        table.drop()
        while blocks:
            blocks.pop()
        torch.cuda.empty_cache()
        size = int((meminfo()["MemAvailable"] - args.held * 2**30) // (1 << 30))
        for _ in range(max(0, size)):
            b = torch.empty((1 << 30,), dtype=torch.uint8, device="cuda")
            b.fill_(1)
            blocks.append(b)
        torch.cuda.synchronize()
        m = meminfo()
        return {"held_gib": len(blocks), "MemAvailable_gib": gib(m["MemAvailable"]), "MemFree_gib": gib(m["MemFree"]),
                "resident": table.resident()}

    options = (("default", False, False, None), ("threads64", False, False, 64), ("rows_ahead", False, True, None),
               ("stage_ahead", True, False, None), ("stage_ahead+rows_ahead", True, True, None),
               ("stage_ahead+threads64", True, False, 64), ("default again", False, False, None))
    for name, stage, rows, threads in options:
        STATE["stage_ahead"], STATE["rows_ahead"] = stage, rows
        set_threads(table, threads)
        state = hold()
        run(engine, nat, f"held cold {name} natural", table, runs, {"held": state})
    STATE["stage_ahead"] = STATE["rows_ahead"] = False
    set_threads(table, None)
    state = hold()
    run(engine, syn, "held cold default synthetic", table, runs, {"held": state})
    STATE["stage_ahead"] = True
    state = hold()
    run(engine, syn, "held cold stage_ahead synthetic", table, runs, {"held": state})
    STATE["stage_ahead"] = False
    while blocks:
        blocks.pop()
    torch.cuda.empty_cache()
    forget(engine)
    hashes: dict = {}
    for r in runs:
        hashes.setdefault(r["label"].split()[-1], set()).add(r["logits"])
    report["identical_logits"] = {k: len(v) == 1 for k, v in hashes.items()}
    passed = sum(1 for v in report["identical_logits"].values() if v)
    (args.out / "residency_held.json").write_text(json.dumps(report, indent=1, default=str))
    (args.out / "ledger_metrics.json").write_text(json.dumps({
        "prefill_32k": next(r["tok_s"] for r in runs if r["label"] == "unheld warm default synthetic"),
        "tests_passed": passed, "tests_failed": len(report["identical_logits"]) - passed}))
    print(f"[ngram] identical head logits per prompt across options: {report['identical_logits']}", flush=True)
    print(f"[ngram] done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
