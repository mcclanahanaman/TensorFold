"""Flash Next's decode verification DeltaNet windows on Jetson Thor: gdn.cu's tree kernel against its chain-kernel
variants (DPUSH-5), times and bits.

    python tools/thor/gdn_tree_bench.py OUT.json [--streams 1,4,8] [--windows 2,4,7] [--pending 0,1,4,7]
        [--variants 0,1,2] [--layers 6] [--iters 20] [--check] [--res]

Decode shapes: 16 key heads (fp32 q and k, L2-normed, q times 128^-0.5), 48 value heads of 128 (bf16 v), per-head
decays g in (0.69, 0.99) and beta = bf16(sigmoid), an fp32 state [48, 128, 128] a stream. Every stream's window is a
chain of ``w`` rows (no slots) after ``p`` pending rows folded into its state in place, called as gdn_multi.block
calls it (a table of state pointers, starts, pending k, v, g, beta, rows and counts); p = 0 passes no pending rows.
Variant v runs with TENSORFOLD_THOR_GDN_CHAIN=v (0: the tree kernel). Each variant's launches over ``layers`` sets of
states (more bytes than the L2 holds) are captured in one CUDA graph and replayed ``iters`` times after a warm-up spin;
time = CUDA events / launches; GB/s = state bytes read, plus written when p > 0, over the time. ``--check``: each
variant's outputs and committed states equal variant 0's bit for bit from the same states. ``--res``: registers,
stack and spills of the built tree and chain kernels (cuobjdump -res-usage on the extension).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import torch

HK, HV, DV, DK = 16, 48, 128, 128


def ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def rows_inputs(rows: int, seed: int) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def normed(x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True)

    q = (normed(torch.randn((rows, HK, DK), generator=gen, device="cuda")) * DK ** -0.5).contiguous()
    k = normed(torch.randn((rows, HK, DK), generator=gen, device="cuda")).contiguous()
    v = torch.randn((rows, HV, DV), generator=gen, device="cuda").bfloat16()
    g = torch.rand((rows, HV), generator=gen, device="cuda") * 0.3 + 0.69
    beta = torch.sigmoid(torch.randn((rows, HV), generator=gen, device="cuda")).bfloat16().float()
    return q, k, v, g, beta


class Case:
    """One (streams, window, pending) shape: inputs, a plan, pending rows and ``layers`` sets of states."""

    def __init__(self, gdn, streams: int, window: int, pending: int, layers: int, seed: int) -> None:
        self.s, self.w, self.p = streams, window, pending
        self.x = rows_inputs(streams * window, seed)
        self.plan = gdn.plan([list(range(-1, window - 1))] * streams, "cuda")
        gen = torch.Generator(device="cuda").manual_seed(seed + 7)
        self.base = [torch.randn((streams, HV, DV, DK), generator=gen, device="cuda") * 0.05 for _ in range(layers)]
        self.states = [b.clone() for b in self.base]
        self.tables = [self.table(st) for st in self.states]
        self.pend = None
        if pending:
            pk, pv, pg, pb = rows_inputs(streams * pending, seed + 3)[1:]
            rows = torch.arange(streams * pending, dtype=torch.int32, device="cuda").view(streams, pending)
            counts = torch.full((streams,), pending, dtype=torch.int32, device="cuda")
            self.pend = (pk, pv, pg, pb, rows, counts)

    @staticmethod
    def table(st: torch.Tensor) -> torch.Tensor:
        return torch.tensor([st[i].data_ptr() for i in range(st.shape[0])], dtype=torch.int64, device="cuda")

    def call(self, gdn, layer: int) -> torch.Tensor:
        return gdn.tree(*self.x, self.plan, table=self.tables[layer], pending=self.pend)

    def state_bytes(self) -> int:
        return self.s * HV * DV * DK * 4 * (2 if self.p else 1)


def spin(seconds: float) -> None:
    a = torch.randn((2048, 2048), device="cuda")
    t = time.time()
    while time.time() - t < seconds:
        a = (a @ a).clamp_(-1, 1)
    torch.cuda.synchronize()


def timed(gdn, case: Case, iters: int) -> float:
    """us a launch: the case's launches over its layers in one graph, replayed ``iters`` times."""

    for li in range(len(case.states)):
        case.call(gdn, li)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for li in range(len(case.states)):
            case.call(gdn, li)
    graph.replay()
    torch.cuda.synchronize()
    return replays(graph, iters, len(case.states))


def replays(graph, iters: int, launches: int, reps: int = 3) -> float:
    """us a launch, the median of ``reps`` timings of ``iters`` replays."""

    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            graph.replay()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) * 1e3 / (iters * launches))
    return sorted(times)[len(times) // 2]


def floors(streams: int, layers: int, iters: int) -> dict:
    """The state traffic alone: an in-place multiply (read + write) and a sum (read) over each layer's states."""

    xs = [torch.randn((streams, HV, DV, DK), device="cuda") for _ in range(layers)]
    out = {}
    for name, op in (("rw", lambda x: x.mul_(1.0)), ("r", lambda x: x.sum())):
        for x in xs:
            op(x)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for x in xs:
                op(x)
        us = replays(graph, iters, layers)
        nbytes = xs[0].numel() * 4 * (2 if name == "rw" else 1)
        out[name] = {"us": round(us, 2), "GBps": round(nbytes / us / 1e3, 1)}
    return out


def run_once(gdn, case: Case) -> tuple[torch.Tensor, torch.Tensor]:
    case.states[0].copy_(case.base[0])
    y = case.call(gdn, 0)
    torch.cuda.synchronize()
    return y.clone(), case.states[0].clone()


def res_usage(gdn) -> list[str]:
    path = gdn._ext().__file__
    try:
        out = subprocess.run(["cuobjdump", "-res-usage", path], capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError) as e:
        return [f"cuobjdump failed: {e}"]
    lines, name = [], None
    for line in out.splitlines():
        if "Function" in line:
            name = line.split("Function", 1)[1].strip(" :")
        elif "REG:" in line and name and "prefill" not in name and any(
                t in name for t in ("chain_kernelIf", "chain_pf_kernelIf", "tree_kernelIfLi0ELi8ELi4ELb1")):
            lines.append(f"{name[30:100]} | {line.strip()[:60]}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--streams", default="1,4,8")
    ap.add_argument("--windows", default="2,4,7")
    ap.add_argument("--pending", default="0,1,4,7")
    ap.add_argument("--variants", default="0,1,2")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--res", action="store_true")
    a = ap.parse_args()
    from tensorfold.cuda.kernels import gdn

    variants = ints(a.variants)
    props = torch.cuda.get_device_properties(0)
    smem = {n: getattr(props, n, None) for n in ("multi_processor_count", "shared_memory_per_multiprocessor",
                                                 "shared_memory_per_block_optin", "L2_cache_size")}
    print("[props]", smem, flush=True)
    result = {"device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()), "props": smem,
              "layers": a.layers, "iters": a.iters, "variants": variants, "cases": []}
    gdn._ext()                                       # build before the spin: Thor's clocks drop while nvcc runs
    if a.res:
        result["res"] = res_usage(gdn)
        for line in result["res"]:
            print("[res]", line)
    spin(1.0)
    result["floors"] = {}
    for s_ in ints(a.streams):
        result["floors"][s_] = f = floors(s_, a.layers, a.iters)
        print(f"[floor] s{s_}: read+write {f['rw']['us']} us ({f['rw']['GBps']} GB/s), read {f['r']['us']} us "
              f"({f['r']['GBps']} GB/s)", flush=True)
    bad = 0
    for s in ints(a.streams):
        for w in ints(a.windows):
            for p in ints(a.pending):
                case = Case(gdn, s, w, p, a.layers, seed=1000 * s + 10 * w + p)
                row = {"streams": s, "window": w, "pending": p, "us": {}, "GBps": {}, "bits": {}}
                ref = None
                for v in variants:
                    os.environ[gdn.CHAIN_ENV] = str(v)
                    if a.check:
                        y, st = run_once(gdn, case)
                        if ref is None:
                            ref = (y, st)
                        same = bool(torch.equal(y.view(torch.int16), ref[0].view(torch.int16)) and
                                    torch.equal(st.view(torch.int32), ref[1].view(torch.int32)))
                        row["bits"][v] = same
                        bad += not same
                    us = timed(gdn, case, a.iters)
                    row["us"][v] = round(us, 2)
                    row["GBps"][v] = round(case.state_bytes() / us / 1e3, 1)
                result["cases"].append(row)
                cells = " ".join(f"v{v} {row['us'][v]:7.1f}" + ("" if row["bits"].get(v, True) else "!BITS") for v in variants)
                print(f"[bench] s{s} w{w} p{p}: {cells} us | best {min(row['GBps'].values()):.0f}-"
                      f"{max(row['GBps'].values()):.0f} GB/s", flush=True)
                del case
                torch.cuda.empty_cache()
    os.environ.pop(gdn.CHAIN_ENV, None)
    result["bit_mismatches"] = bad
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[bench] done: {len(result['cases'])} cases, bit mismatches {bad if a.check else 'unchecked'}")


if __name__ == "__main__":
    main()
