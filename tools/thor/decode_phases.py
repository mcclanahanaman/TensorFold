"""A concurrent decode round's phases on Jetson Thor, timed in process (FR-OPS-17; THOR-DEC-1).

    python tools/thor/decode_phases.py OUT.json --model MODEL [--label L] [--users 1,4] [--types prose,code]
        [--groups 2] [--new 400] [--nsys u1:20:40,u4:20:40] [--sparkdash tools/thor/_ext/decode0/sparkdash_port.py]
        [--cpu] [--sched auto|spin|yield|blocking]

The engine is built as ``tensorfold serve MODEL --backend cuda --kv-dtype int8 --parallel 4`` builds it
(``prefill_profile.build_engine``; the Thor switches come from the environment). The requests are sparkDash's decode
prompts (``pick_decode_bench_prompts`` of the stdlib port: the `` (stream i/n)`` suffix at 2+ streams, one code task a
stream), rendered with thinking off, greedy, end tokens ignored, MTP drafts on, ``--new`` tokens each (sparkDash: 400);
each group starts with every kept prompt end dropped.

Per decode round (every stream has its first token) the wall time and, by wrapping ``multi.py``'s calls (kernels and
launches untouched), each phase's host time and its GPU span between CUDA events on the current stream:

  stage    the verification windows staged          tables  attn_multi.Step + gdn_multi.Tables
  verify   the verification forward (eager, a graph replay, or a recording and its replay)
  sample   sample_streams with its read-back          commit  accept + commit + gdn keep (host loop and launches)
  mtp_stage, mtp, picks   per MTP step: staging, the forward, the read-back of its picks

``gap`` is the round's GPU span less its phases' spans: where the GPU waited on the host between phases. Graph
counters (forwards, replays, captures, capture seconds) are read per round from ``MultiDecoder.graphs``. ``--nsys``
brackets decode rounds [A, B) of the first group of U users with cudaProfilerStart/Stop (after a device sync), and
every phase is an NVTX range, so ``nsys --capture-range=cudaProfilerApi`` attributes kernels to phases.

``--cpu`` samples the host CPU at a round's start and right after the sample and first picks read-backs: the CPU the
thread runs on, its ``scaling_cur_freq`` and a fixed pure-Python loop's time (the host's effective speed); the
report's ``cpu`` holds the governor, frequency limits and idle states. ``--sched`` sets the CUDA primary context's
scheduling flag before torch starts CUDA (the driver's wait in a synchronize: spin, yield or blocking sleep).
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PHASES = ("stage", "tables", "verify", "sample", "commit", "mtp_stage", "mtp", "picks")
SCHED = {"auto": 0, "spin": 1, "yield": 2, "blocking": 4}
PROBE_N = 2000                                      # the CPU probe's loop: ~50 us at full clock
SYS_CPU = Path("/sys/devices/system/cpu")


def _read(p: Path) -> str:
    try:
        return p.read_text().strip()
    except OSError as exc:
        return f"<{type(exc).__name__}>"


def cpu_info() -> dict:
    """The host CPUs' frequency policy and idle states, read once."""

    keys = ("scaling_driver", "scaling_governor", "scaling_min_freq", "scaling_max_freq", "cpuinfo_min_freq",
            "cpuinfo_max_freq", "scaling_cur_freq")
    cpus = sorted((d for d in SYS_CPU.glob("cpu[0-9]*")), key=lambda d: int(d.name[3:]))
    idle = sorted((SYS_CPU / "cpu0" / "cpuidle").glob("state*"))
    return {"online": _read(SYS_CPU / "online"), "affinity": sorted(os.sched_getaffinity(0)),
            "cpus": [{"cpu": int(d.name[3:]), **{k: _read(d / "cpufreq" / k) for k in keys}} for d in cpus],
            "idle": [{k: _read(s / k) for k in ("name", "latency", "residency", "disable")} for s in idle]}


class Cpu:
    """The calling thread's CPU, that CPU's current frequency and a fixed loop's time, at a moment of the round."""

    def __init__(self) -> None:
        self.libc = ctypes.CDLL(None)
        self.fds: dict[int, int] = {}
        self.rows: list = []

    def sample(self, at: str) -> None:
        t = time.perf_counter_ns()
        x = 0
        for i in range(PROBE_N):
            x += i
        probe = (time.perf_counter_ns() - t) / 1e3
        cpu = int(self.libc.sched_getcpu())
        fd = self.fds.get(cpu)
        if fd is None:
            try:
                fd = os.open(SYS_CPU / f"cpu{cpu}" / "cpufreq" / "scaling_cur_freq", os.O_RDONLY)
            except OSError:
                fd = -1
            self.fds[cpu] = fd
        try:
            khz = int(os.pread(fd, 32, 0)) if fd >= 0 else 0
        except (OSError, ValueError):
            khz = 0
        self.rows.append((at, cpu, khz, probe))

    def take(self) -> list:
        out, self.rows = self.rows, []
        return out


def _cu():
    return ctypes.CDLL("libcuda.so.1")


def set_sched(name: str) -> dict:
    """The primary context's scheduling flag, set before torch creates the context."""

    cu = _cu()
    dev = ctypes.c_int()
    rc = [cu.cuInit(0), cu.cuDeviceGet(ctypes.byref(dev), 0)]
    fn = getattr(cu, "cuDevicePrimaryCtxSetFlags_v2", None) or cu.cuDevicePrimaryCtxSetFlags
    rc.append(fn(dev, SCHED[name]))
    return {"set": name, "rc": rc}


def ctx_state() -> dict:
    cu = _cu()
    dev, flags, active = ctypes.c_int(), ctypes.c_uint(), ctypes.c_int()
    rc = [cu.cuDeviceGet(ctypes.byref(dev), 0),
          cu.cuDevicePrimaryCtxGetState(dev, ctypes.byref(flags), ctypes.byref(active))]
    return {"flags": flags.value, "active": active.value, "rc": rc}


class Timer:
    """Host and GPU time of each wrapped call, collected per round."""

    def __init__(self) -> None:
        self.depth = 0
        self.rows: list = []          # (phase, host seconds, start event, end event)
        self.hooks: dict = {}         # phase -> callable(phase), run after the call (outside its timing)
        self.hooked: set = set()      # phases already hooked this round (first call only)

    def wrap(self, phase: str, fn):
        def timed(*args, **kwargs):
            if self.depth:                                # nested (a forward inside a timed graph run): untimed
                return fn(*args, **kwargs)
            self.depth += 1
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.nvtx.range_push(phase)
            e0.record()
            t = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                host = time.perf_counter() - t
                e1.record()
                torch.cuda.nvtx.range_pop()
                self.depth -= 1
                self.rows.append((phase, host, e0, e1))
                hook = self.hooks.get(phase)
                if hook is not None and phase not in self.hooked:
                    self.hooked.add(phase)
                    hook(phase)
        return timed

    def take(self) -> dict:
        """This round's phases (call after a device sync): {phase: [host ms, gpu ms, calls]}."""

        out: dict = {}
        for phase, host, e0, e1 in self.rows:
            got = out.setdefault(phase, [0.0, 0.0, 0])
            got[0] += host * 1e3
            got[1] += e0.elapsed_time(e1)
            got[2] += 1
        self.rows = []
        self.hooked = set()
        return out


def install(timer: Timer) -> None:
    from tensorfold.families.qwen4_exp.cuda import attn_multi, gdn_multi, multi, thor_graphs

    multi.stage = timer.wrap("stage", multi.stage)
    multi.compute = timer.wrap("verify", multi.compute)               # eager verification (graphs off or eager key)
    multi.sample_streams = timer.wrap("sample", multi.sample_streams)
    multi.mtp_stage = timer.wrap("mtp_stage", multi.mtp_stage)
    step_cls, tables_cls = attn_multi.Step, gdn_multi.Tables
    attn_multi.Step = timer.wrap("tables", step_cls)
    gdn_multi.Tables = timer.wrap("tables", tables_cls)
    md = multi.MultiDecoder
    md._mtp = timer.wrap("mtp", md._mtp)
    md._picks = timer.wrap("picks", md._picks)
    run = thor_graphs.DecodeGraphs.run

    def graphs_run(self, kind, *args, **kwargs):
        if kind == "verify":
            return timer.wrap("verify", run)(self, kind, *args, **kwargs)
        return run(self, kind, *args, **kwargs)                       # MTP: inside the timed _mtp
    thor_graphs.DecodeGraphs.run = graphs_run
    commit, accept, keep = multi.commit, multi.accept, gdn_multi.keep
    multi.commit = timer.wrap("commit", commit)
    multi.accept = timer.wrap("commit", accept)
    gdn_multi.keep = timer.wrap("commit", keep)


def graph_counts(multi) -> dict:
    g = getattr(multi, "graphs", None)
    if g is None:
        return {}
    return {"forwards": sum(g.forwards.values()), "replays": sum(g.replays.values()),
            "captures": sum(g.captures.values()), "capture_s": g.capture_s, "graphs": len(g.graphs),
            "graph_mib": g.graph_mib}


def mean(v):
    return statistics.fmean(v) if v else 0.0


def summarize(rounds: list) -> dict:
    """Means over a group's decode rounds."""

    out = {"rounds": len(rounds), "wall_ms": mean([r["wall_ms"] for r in rounds]),
           "wall_p50_ms": statistics.median([r["wall_ms"] for r in rounds]) if rounds else 0.0,
           "span_ms": mean([r["span_ms"] for r in rounds]), "gap_ms": mean([r["gap_ms"] for r in rounds]),
           "rows": mean([r["rows"] for r in rounds]), "mtp_steps": mean([r["phases"].get("mtp", [0, 0, 0])[2]
                                                                         for r in rounds])}
    for p in PHASES:
        out[p] = {"host_ms": mean([r["phases"].get(p, [0, 0, 0])[0] for r in rounds]),
                  "gpu_ms": mean([r["phases"].get(p, [0, 0, 0])[1] for r in rounds])}
    for k in ("replays", "captures", "capture_s"):
        out[k] = sum(r.get("graph", {}).get(k, 0) for r in rounds)
    at = {}
    for r in rounds:
        for name, cpu, khz, probe in r.get("cpu", []):
            got = at.setdefault(name, {"khz": [], "probe_us": [], "cpus": set()})
            got["khz"].append(khz)
            got["probe_us"].append(probe)
            got["cpus"].add(cpu)
    if at:
        out["cpu"] = {name: {"khz": mean(v["khz"]), "khz_min": min(v["khz"]), "probe_us": mean(v["probe_us"]),
                             "probe_p50_us": statistics.median(v["probe_us"]), "cpus": sorted(v["cpus"]),
                             "n": len(v["khz"])} for name, v in at.items()}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--users", default="1,4")
    ap.add_argument("--types", default="prose,code")
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--new", type=int, default=400)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--kv-dtype", default="int8")
    ap.add_argument("--nsys", default="")
    ap.add_argument("--sparkdash", type=Path, default=HERE / "_ext" / "decode0" / "sparkdash_port.py")
    ap.add_argument("--cpu", action="store_true", help="sample the host CPU at round start and after read-backs")
    ap.add_argument("--sched", choices=sorted(SCHED), default=None)
    a = ap.parse_args(argv)
    sched = set_sched(a.sched) if a.sched else None

    spec = importlib.util.spec_from_file_location("sparkdash_port", a.sparkdash)
    sd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sd)
    import ngram_residency as nr
    import prefill_profile as pp
    from tokenizers import Tokenizer

    from tensorfold.cuda.chat_template import ChatTemplate
    from tensorfold.cuda.streams import Stream

    t0 = time.perf_counter()
    engine = pp.build_engine(a.model, a.parallel, a.kv_dtype)
    print(f"[phases] engine up in {time.perf_counter() - t0:.0f}s", flush=True)
    host = {"sched": sched, "ctx": ctx_state(), "spin": os.environ.get("TENSORFOLD_THOR_DECODE_SPIN", ""),
            **cpu_info()}
    c0 = host["cpus"][0] if host["cpus"] else {}
    print(f"[phases] host: ctx {host['ctx']} sched {sched} spin {host['spin']!r} governor "
          f"{c0.get('scaling_governor')} driver {c0.get('scaling_driver')} freq {c0.get('scaling_min_freq')}-"
          f"{c0.get('scaling_max_freq')} kHz, idle {[s['name'] for s in host['idle']]}", flush=True)
    tok = Tokenizer.from_file(str(a.model / "tokenizer.json"))
    template = ChatTemplate(a.model)

    def ids_of(text: str) -> list[int]:
        rendered = template.render([{"role": "user", "content": text}], tools=None, enable_thinking=False)
        return tok.encode(rendered, add_special_tokens=False).ids

    windows = {}
    for w in [w for w in a.nsys.split(",") if w]:
        u, lo, hi = w.split(":")
        windows[int(u.lstrip("u"))] = (int(lo), int(hi))
    timer = Timer()
    install(timer)
    cpu = Cpu() if a.cpu else None
    if cpu is not None:
        timer.hooks = {"sample": cpu.sample, "picks": cpu.sample}
    multi = engine.multi
    report = {"label": a.label, "model": str(a.model), "new": a.new, "host": host, "groups": []}
    profiling = False
    for users in [int(u) for u in a.users.split(",") if u]:
        for kind in [t for t in a.types.split(",") if t]:
            prompts = sd.pick_decode_bench_prompts(users, kind)
            for g in range(a.groups):
                nr.forget(engine)
                streams = [Stream(ids_of(p), a.new, None, draft=True, stop_eos=False) for p in prompts]
                win = windows.get(users) if (g == 0 and kind == a.types.split(",")[0]) else None
                torch.cuda.synchronize()
                for s in streams:
                    multi.admit(s)
                rounds, r, fill_s = [], 0, 0.0
                while not all(s.done for s in streams):
                    decoding = all(len(s.out) >= 1 for s in streams)
                    if win and decoding and r == win[0] and not profiling:
                        torch.cuda.synchronize()
                        torch.cuda.profiler.start()
                        profiling = True
                    before = graph_counts(multi)
                    rows = sum(1 + len(s.drafts) for s in streams if not s.done and len(s.out) >= 1)
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    torch.cuda.nvtx.range_push(f"round u{users}" if decoding else f"fill u{users}")
                    if cpu is not None and decoding:
                        cpu.sample("start")
                    e0.record()
                    t = time.perf_counter()
                    done = multi.round()
                    multi.finish(done)
                    e1.record()
                    torch.cuda.synchronize()
                    wall = time.perf_counter() - t
                    torch.cuda.nvtx.range_pop()
                    phases = timer.take()
                    taken = cpu.take() if cpu is not None else []
                    if not decoding:
                        fill_s += wall
                        continue
                    after = graph_counts(multi)
                    span = e0.elapsed_time(e1)
                    rounds.append({"wall_ms": wall * 1e3, "span_ms": span,
                                   "gap_ms": span - sum(v[1] for v in phases.values()), "phases": phases,
                                   "rows": rows, "cpu": taken,
                                   "graph": {k: after[k] - before.get(k, 0) for k in after}})
                    r += 1
                    if profiling and win and r >= win[1]:
                        torch.cuda.synchronize()
                        torch.cuda.profiler.stop()
                        profiling = False
                if profiling:
                    torch.cuda.synchronize()
                    torch.cuda.profiler.stop()
                    profiling = False
                outs = [len(s.out) for s in streams]
                tokens = sum(max(0, o - 1) for o in outs)
                dec_s = sum(x["wall_ms"] for x in rounds) / 1e3
                row = {"users": users, "type": kind, "group": g, "prompt_tokens": [len(s.prompt) for s in streams],
                       "out": outs, "errors": [repr(s.error) for s in streams if s.error is not None],
                       "fill_s": fill_s, "decode_tokens": tokens, "decode_s": dec_s,
                       "agg_tok_s": tokens / dec_s if dec_s else 0.0,
                       "tokens_per_round": tokens / len(rounds) / users if rounds else 0.0,
                       "drafted": sum(s.drafted for s in streams), "accepted": sum(s.accepted for s in streams),
                       "summary": summarize(rounds), "graph_total": graph_counts(multi), "rounds": rounds,
                       "profiled": bool(win)}
                report["groups"].append(row)
                sm = row["summary"]
                print(f"[phases] {a.label} u{users} {kind} g{g}: {row['agg_tok_s']:.1f} tok/s, "
                      f"{row['tokens_per_round']:.2f} tok/round/user, round {sm['wall_ms']:.2f} ms (p50 "
                      f"{sm['wall_p50_ms']:.2f}), verify gpu {sm['verify']['gpu_ms']:.2f} host "
                      f"{sm['verify']['host_ms']:.2f}, mtp gpu {sm['mtp']['gpu_ms']:.2f} x{sm['mtp_steps']:.2f}, "
                      f"gap {sm['gap_ms']:.2f}, captures {sm['captures']} ({sm['capture_s']:.2f} s), "
                      f"replays {sm['replays']}", flush=True)
                for name, c in sm.get("cpu", {}).items():
                    print(f"[phases]   cpu@{name}: {c['khz'] / 1e3:.0f} MHz (min {c['khz_min'] / 1e3:.0f}), probe "
                          f"{c['probe_us']:.1f} us (p50 {c['probe_p50_us']:.1f}), cpus {c['cpus']}", flush=True)
                a.out.write_text(json.dumps(report))
    a.out.write_text(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
