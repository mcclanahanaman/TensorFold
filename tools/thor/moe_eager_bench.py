"""Expert calls as TensorFold's concurrent server issues them (eager: no CUDA graphs), on one real layer: the host's
time to issue a call, the wall time a call costs back to back, and the kernels a call launches with their GPU time,
TensorFold's portable NVFP4 path against the Thor path (tensorfold.cuda.nvfp4.thor_experts).

    FLASHINFER_NO_DOWNLOAD=1 python tools/thor/moe_eager_bench.py --model /hf/... --out /out/moe_eager.json

A call is the part that differs between the paths: route (each path's tile) and the routed experts into the slots
(the router, the shared expert and the combine are the same on both). The Thor path runs twice: launching its kernels
every call (``thor_launch``) and, where the code records eager decode calls, replaying its record (``thor``: one CUDA
graph a layer and row count). ``decode`` calls use the prompt plan and bf16 slots (TensorFold 0.6.0 decodes with its
prompt arithmetic); ``mtp`` calls the decode plan and fp32 slots (the MTP head's drafts).

Host: ``--calls`` calls issued while a GPU sleep holds the queue (nothing completes meanwhile, so the loop times the
host alone; a call that synchronized would wait out the sleep), with its FlashInfer, Triton-glue and route shares by
wrappers. Wall: the same calls back to back, synchronized once (CUDA events alongside). Kernels: torch.profiler's GPU
activities a call.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import moe_common as mc  # noqa: E402

SLEEP_CYCLES = 1_000_000_000          # a GPU spin longer than any host loop here (about 0.5-1 s on Thor)


class Shares:
    """Host seconds spent inside the wrapped launch paths (FlashInfer's GEMM call, the Triton glue, the route)."""

    def __init__(self) -> None:
        self.t = {"fi": 0.0, "glue": 0.0, "route": 0.0}

    def wrap(self, key: str, fn):
        def go(*a, **k):
            t = time.perf_counter()
            r = fn(*a, **k)
            self.t[key] += time.perf_counter() - t
            return r
        return go


class TimedKernel:
    def __init__(self, kernel, shares: Shares) -> None:
        self.kernel, self.shares = kernel, shares

    def __getitem__(self, grid):
        return self.shares.wrap("glue", self.kernel[grid])


class Call:
    """One path's route + routed experts at m rows on fixed buffers (as a server's buffer set holds them)."""

    def __init__(self, ex, thor_path: bool, m: int, mtp: bool, x: torch.Tensor, picks: torch.Tensor, shares: Shares,
                 records: bool = True):
        from tensorfold.cuda import experts as grouped
        from tensorfold.cuda.nvfp4 import experts as nvx
        from tensorfold.cuda.nvfp4 import thor_experts as thor

        self.grouped, self.nvx, self.thor, self.shares = grouped, nvx, thor, shares
        self.ex, self.thor_path, self.m = ex, thor_path, m
        self.x, self.picks = x, picks
        self.plan = grouped.Plan(m, mc.TOP + 1, mc.E + 1, "cuda", prefill=not mtp)
        self.y = torch.zeros(m, mc.TOP + 1, mc.D, dtype=torch.float32 if mtp else torch.bfloat16, device="cuda")
        if thor_path:
            self.tile = thor.tactic(m)[0]
            self.scr = thor.Scratch(m, mc.TOP, mc.E, mc.NI, mc.D, "cuda")
            if not records and getattr(self.scr, "graphs", None) is not None:
                self.scr.graphs = None              # every call launches its kernels (the path before its records)
        else:
            self.tile = nvx.PREFILL_TILE
            self.act = torch.zeros(m * (mc.TOP + 1), mc.NI, dtype=torch.bfloat16, device="cuda")

    def __call__(self) -> None:
        t = time.perf_counter()
        self.grouped.route(self.picks, self.plan, self.tile)
        self.shares.t["route"] += time.perf_counter() - t
        if self.thor_path:
            self.thor.run(self.x, self.ex, self.plan, self.y, self.m, mc.TOP, self.scr)
        else:
            self.nvx.gate_up(self.x, self.ex, self.plan, self.act, self.m, skip=mc.E)
            self.nvx.down(self.act, self.ex, self.plan, self.y.view(-1, mc.D), self.m, skip=mc.E)


def host_us(fn, calls: int, shares: Shares) -> dict:
    torch.cuda.synchronize()
    for k in shares.t:
        shares.t[k] = 0.0
    torch.cuda._sleep(SLEEP_CYCLES)
    t = time.perf_counter()
    for _ in range(calls):
        fn()
    dt = time.perf_counter() - t
    t = time.perf_counter()
    torch.cuda.synchronize()
    waited = time.perf_counter() - t
    out = {"host_us": dt / calls * 1e6, "sleep_left_s": waited}
    out.update({f"{k}_us": v / calls * 1e6 for k, v in shares.t.items()})
    return out


def wall_us(fn, calls: int) -> dict:
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t = time.perf_counter()
    a.record()
    for _ in range(calls):
        fn()
    b.record()
    b.synchronize()
    return {"wall_us": (time.perf_counter() - t) / calls * 1e6, "event_us": a.elapsed_time(b) / calls * 1e3}


def kernels(fn, calls: int) -> dict:
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            fn()
        torch.cuda.synchronize()
    gpu = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    names: dict[str, int] = {}
    for e in gpu:
        names[e.name[:70]] = names.get(e.name[:70], 0) + 1
    busy = sum(e.time_range.elapsed_us() for e in gpu)
    return {"gpu_ops": len(gpu) / calls, "gpu_busy_us": busy / calls,
            "names": {k: v / calls for k, v in sorted(names.items(), key=lambda kv: -kv[1])}}


def mem_available() -> int:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def records_cost(ex, router, rows_max: int, copies: int) -> dict:
    """What the Thor path's records of eager decode calls cost: a first call (launch + record) and a replay, per row
    count, and the memory ``copies`` x ``rows_max`` records hold (``copies`` output buffers stand in for layers)."""

    import gc

    from tensorfold.cuda import experts as grouped
    from tensorfold.cuda.nvfp4 import thor_experts as thor

    scr = thor.Scratch(rows_max, mc.TOP, mc.E, mc.NI, mc.D, "cuda")
    if getattr(scr, "graphs", None) is None:
        return {"records": "this code keeps none"}
    plan = grouped.Plan(rows_max, mc.TOP + 1, mc.E + 1, "cuda", prefill=True)
    x = mc.activations(rows_max, seed=5)
    ids, _ = mc.route(x, router)
    picks = torch.cat([ids, torch.full((rows_max, 1), mc.E, dtype=torch.int32, device="cuda")], 1).contiguous()
    ys = [torch.zeros(rows_max, mc.TOP + 1, mc.D, dtype=torch.bfloat16, device="cuda") for _ in range(copies)]

    def call(n: int, y: torch.Tensor) -> float:
        grouped.route(picks[:n], plan, thor.tactic(n)[0])
        torch.cuda.synchronize()
        t = time.perf_counter()
        thor.run(x[:n], ex, plan, y[:n], n, mc.TOP, scr)
        torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1e3

    gc.collect()
    torch.cuda.synchronize()
    time.sleep(1)
    avail0, free0 = mem_available(), torch.cuda.mem_get_info()[0]
    first = [call(n, ys[0]) for n in range(1, rows_max + 1)]
    for y in ys[1:]:
        for n in range(1, rows_max + 1):
            call(n, y)
    torch.cuda.synchronize()
    time.sleep(1)
    avail1, free1 = mem_available(), torch.cuda.mem_get_info()[0]
    replay = [call(n, ys[0]) for n in range(1, rows_max + 1)]
    count = len(scr.graphs)
    return {"records": count, "first_call_ms": first, "replay_ms": replay,
            "mem_available_drop_mib": (avail0 - avail1) / 2**20, "cuda_free_drop_mib": (free0 - free1) / 2**20,
            "per_record_kib": (avail0 - avail1) / max(count, 1) / 1024}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="/out/moe_eager.json")
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--decode-ms", default="1,2,3,4,5,7,14,28")
    ap.add_argument("--mtp-ms", default="1,2,4,7,28")
    ap.add_argument("--calls", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--record-copies", type=int, default=10, help="output buffers standing in for layers (0: skip)")
    args = ap.parse_args()

    from tensorfold.cuda.nvfp4 import experts as nvx
    from tensorfold.cuda.nvfp4 import thor_experts as thor

    t0 = time.time()
    layer = mc.Checkpoint(args.model).layer(args.layer)
    p = lambda q: (q.w, q.s, q.s2)  # noqa: E731
    exs = {"tf": nvx.make(p(layer.gate), p(layer.up), p(layer.down)),
           "thor": thor.make(p(layer.gate), p(layer.up), p(layer.down))}
    print(f"[{mc.stamp()}] layer {args.layer} in both layouts in {time.time() - t0:.0f}s", flush=True)
    shares = Shares()
    fw = thor._fi()
    gemm = fw._run_grouped_gemm
    fw._run_grouped_gemm = shares.wrap("fi", gemm)
    glue = thor._glue
    thor._glue = lambda: tuple(TimedKernel(k, shares) for k in glue())
    report = {"layer": args.layer, "calls": args.calls, "rounds": args.rounds, "rows": []}
    cases = [("decode", int(m)) for m in args.decode_ms.split(",")] + [("mtp", int(m)) for m in args.mtp_ms.split(",")]
    for kind, m in cases:
        x = mc.activations(m, seed=m)
        ids, _ = mc.route(x, layer.router)
        picks = torch.cat([ids, torch.full((m, 1), mc.E, dtype=torch.int32, device="cuda")], 1).contiguous()
        calls = {"tf": Call(exs["tf"], False, m, kind == "mtp", x, picks, shares),
                 "thor_launch": Call(exs["thor"], True, m, kind == "mtp", x, picks, shares, records=False)}
        if hasattr(thor.Scratch(1, mc.TOP, mc.E, mc.NI, mc.D, "cuda"), "graphs"):
            calls["thor"] = Call(exs["thor"], True, m, kind == "mtp", x, picks, shares)
        for c in calls.values():                       # build every kernel this shape reaches
            for _ in range(3):
                c()
        torch.cuda.synchronize()
        row = {"kind": kind, "m": m}
        for name, c in calls.items():
            hosts, walls = [], []
            for _ in range(args.rounds):              # interleaved with the other path's rounds below
                hosts.append(host_us(c, args.calls, shares))
                walls.append(wall_us(c, args.calls))
            med = {k: statistics.median(h[k] for h in hosts) for k in hosts[0]}
            med.update({k: statistics.median(w[k] for w in walls) for k in walls[0]})
            med.update(kernels(c, 10))
            row[name] = med
        report["rows"].append(row)
        for name in calls:
            r = row[name]
            print(f"[{mc.stamp()}] {kind:6s} M={m:3d} {name:11s} host {r['host_us']:7.1f} us (fi {r['fi_us']:6.1f} "
                  f"glue {r['glue_us']:6.1f} route {r['route_us']:5.1f}) | wall {r['wall_us']:7.1f} us | gpu busy "
                  f"{r['gpu_busy_us']:7.1f} us in {r['gpu_ops']:.1f} ops | sleep left {r['sleep_left_s']:.2f}s",
                  flush=True)
        if "thor" in calls:                         # the records replay the launched kernels: the same bits
            row["records_equal_launches"] = bool(torch.equal(calls["thor"].y, calls["thor_launch"].y))
        Path(args.out).write_text(json.dumps(report, indent=1))
    fw._run_grouped_gemm = gemm
    thor._glue = glue
    if args.record_copies:
        report["records_cost"] = rc = records_cost(exs["thor"], layer.router, 28, args.record_copies)
        if isinstance(rc.get("records"), int):
            print(f"[{mc.stamp()}] records: {rc['records']} hold {rc['mem_available_drop_mib']:.1f} MiB of "
                  f"MemAvailable ({rc['per_record_kib']:.1f} KiB each; cuda free {rc['cuda_free_drop_mib']:.1f} MiB); "
                  f"first call ms {[round(v, 2) for v in rc['first_call_ms'][:8]]}...; replay ms "
                  f"{[round(v, 3) for v in rc['replay_ms'][:8]]}...", flush=True)
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
