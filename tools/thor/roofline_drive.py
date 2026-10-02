"""Flash Next on Jetson Thor as served, marked for Nsight Systems and Nsight Compute (FR-OPS-17 roofline-audit).

    python tools/thor/roofline_drive.py prefill OUT --model MODEL [--tokens 8192,32768] [--warm 8192]
        [--capture all|N:P,...] [--inventory]
    python tools/thor/roofline_drive.py decode OUT --model MODEL [--users 1,4] [--groups 2] [--new 256]
        [--capture all|uU:A:B,...]
    python tools/thor/roofline_drive.py ceilings OUT

The engine is built as ``tensorfold serve MODEL --backend cuda --kv-dtype int8 --parallel 4`` builds it (the Thor
switches come from the environment, as in the gates) and requests take the server's path:
``MultiDecoder.admit``, then rounds. The kernels are untouched; the tool adds NVTX ranges ("prompt N" and "pass I"
in prefill, "fill uU" and "round uU" for a round of U users before and after every stream has its first token) and
brackets the chosen part with
cudaProfilerStart/Stop after a device sync, so ``nsys --capture-range=cudaProfilerApi`` and ``ncu --profile-from-start
off`` see only that part:

  prefill  cold prompts (the prefix cache dropped first) after an uncaptured warm-up prompt; ``--capture all`` takes
           every measured prompt, ``32768:8`` only pass 8 of the 32,768-token prompt
  decode   U users at once (greedy, end tokens ignored, MTP drafts on), ``--groups`` groups of U chat prompts one after
           the other; ``--capture all`` takes every group, ``u1:16:18`` decode rounds 16 and 17 of the first
           1-user group
  ceilings same-session achievable rates without the engine: bf16 GEMMs (fp32 accumulation) and DRAM copy / read

Each mode writes OUT/drive.json: wall times, tokens a round, the MTP head's drafted and accepted rows, the model's
dimensions and (``--inventory``) the engine's CUDA tensors by owner path. Summaries: nsys_summary.py, ncu_summary.py.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROMPTS = (   # decode_diag.py's twelve chat prompts (the K4-QSA decode diagnosis)
    "Explain how a refrigerator works, step by step.",
    "Write a short story about a robot learning to paint.",
    "Describe the causes and effects of the French Revolution.",
    "What are the main differences between TCP and UDP? Give examples.",
    "Give a detailed recipe for a vegetable lasagna.",
    "Explain the theory of evolution to a high school student.",
    "Write a Python function that merges two sorted lists, and explain it.",
    "Summarize the plot of a classic adventure novel of your choice.",
    "What should someone consider when buying a used car?",
    "Describe how vaccines train the immune system.",
    "Write a letter to a friend about a recent trip to the mountains.",
    "Explain what a hash table is and how collisions are handled.",
)


class Capture:
    """cudaProfilerStart/Stop, each after a device sync so the bubble falls outside the captured part."""

    def __init__(self) -> None:
        self.on = False
        self.windows = 0

    def start(self) -> None:
        if not self.on:
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            self.on, self.windows = True, self.windows + 1

    def stop(self) -> None:
        if self.on:
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            self.on = False


CAP = Capture()


def stats_ms(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    v = sorted(values)
    return {"n": len(v), "mean_ms": 1e3 * statistics.fmean(v), "p50_ms": 1e3 * v[len(v) // 2],
            "p10_ms": 1e3 * v[len(v) // 10], "p90_ms": 1e3 * v[(9 * len(v)) // 10], "sum_s": sum(v)}


def dims(engine) -> dict:
    cfg = engine.w.cfg
    got = {k: v for k, v in vars(cfg).items() if isinstance(v, (int, float, str, bool))} if hasattr(cfg, "__dict__") \
        else {}
    for k in ("layers", "hidden", "experts", "top_k", "moe_width"):
        got[k] = getattr(cfg, k, None)
    got["prefill_rows"] = getattr(engine.multi, "prefill_rows", None)
    return got


def inventory(engine) -> dict:
    """The engine's CUDA tensors (each storage once) by owner path, and their bytes by path pattern (indices -> N)."""

    seen: set = set()
    storages: set = set()
    rows: list = []

    def walk(obj, path: str, depth: int) -> None:
        if depth > 8 or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda:
                st = obj.untyped_storage()
                if st.data_ptr() not in storages:
                    storages.add(st.data_ptr())
                    rows.append((path, list(obj.shape), str(obj.dtype).replace("torch.", ""), st.nbytes()))
            return
        if isinstance(obj, dict):
            items = [(str(k), v) for k, v in obj.items()]
        elif isinstance(obj, (list, tuple)):
            if len(obj) > 4096:                       # token lists and the like: no weights in there
                return
            items = [(str(i), v) for i, v in enumerate(obj)]
        else:
            if type(obj).__module__.startswith(("torch", "builtins", "numpy", "tokenizers")):
                return
            items = list(getattr(obj, "__dict__", {}).items())
        for k, v in items:
            walk(v, f"{path}.{k}", depth + 1)

    walk(engine.w, "w", 0)
    walk(engine.multi, "multi", 0)
    by: dict = defaultdict(lambda: [0, 0])
    for path, _shape, dtype, nbytes in rows:
        key = re.sub(r"\.\d+(?=\.|$)", ".N", path) + f" [{dtype}]"
        by[key][0] += nbytes
        by[key][1] += 1
    groups = sorted(([k, v[0], v[1]] for k, v in by.items()), key=lambda r: -r[1])
    return {"total_bytes": sum(r[3] for r in rows), "tensors": len(rows), "groups": groups,
            "allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved()}


def windows_prefill(spec: str) -> set:
    if spec in ("all", "none", ""):
        return set()
    return {(int(a), int(b)) for a, b in (w.split(":") for w in spec.split(","))}


def prefill(args) -> dict:
    import ngram_residency as nr
    import prefill_profile as pp
    from tensorfold.families.qwen4_exp.cuda import multi as multi_mod

    t0 = time.perf_counter()
    engine = pp.build_engine(args.model, args.parallel, args.kv_dtype)
    report = {"mode": "prefill", "engine_s": time.perf_counter() - t0, "dims": dims(engine), "prompts": []}
    print(f"[drive] engine up in {report['engine_s']:.0f}s: {report['dims']}", flush=True)
    if args.inventory:
        report["inventory"] = inventory(engine)
    md = multi_mod.MultiDecoder
    orig = md._pass
    chosen = windows_prefill(args.capture)
    state = {"prompt": 0, "index": -1, "walls": []}

    def marked(dec, *a, **kw):
        state["index"] += 1
        cap = (state["prompt"], state["index"]) in chosen
        if cap:
            CAP.start()
        torch.cuda.nvtx.range_push(f"pass {state['index']}")
        t = time.perf_counter()
        try:
            return orig(dec, *a, **kw)
        finally:
            torch.cuda.nvtx.range_pop()
            state["walls"].append(time.perf_counter() - t)
            if cap:
                CAP.stop()

    md._pass = marked
    try:
        plan = ([("warm", args.warm)] if args.warm else []) + [("cold", int(n)) for n in args.tokens.split(",")]
        for i, (kind, n) in enumerate(plan):
            ids = pp.prompt_ids(args.model, n, 4242 + n + (7 if kind == "warm" else 0))
            nr.forget(engine)
            state.update(prompt=n if kind == "cold" else -1, index=-1, walls=[])
            if args.capture == "all" and kind == "cold" and not CAP.on:
                CAP.start()
            torch.cuda.nvtx.range_push(f"prompt {n}")
            got = pp.run_prompt(engine, ids)
            torch.cuda.nvtx.range_pop()
            row = {"kind": kind, "tokens": n, "seconds": got["seconds"], "tok_s": got["tok_s"],
                   "passes": len(state["walls"]), "cpu_pass": stats_ms(state["walls"]), "logits": got["logits_sha"],
                   "error": got["error"]}
            report["prompts"].append(row)
            print(f"[drive] {kind} {n}: {got['seconds']:.2f}s = {got['tok_s']:.0f} tok/s, {row['passes']} passes",
                  flush=True)
        CAP.stop()
    finally:
        md._pass = orig
    report["capture_windows"] = CAP.windows
    return report


def decode(args) -> dict:
    import ngram_residency as nr
    import prefill_profile as pp
    from tokenizers import Tokenizer

    from tensorfold.cuda.chat_template import ChatTemplate
    from tensorfold.cuda.streams import Stream

    t0 = time.perf_counter()
    engine = pp.build_engine(args.model, args.parallel, args.kv_dtype)
    report = {"mode": "decode", "engine_s": time.perf_counter() - t0, "dims": dims(engine), "groups": []}
    print(f"[drive] engine up in {report['engine_s']:.0f}s", flush=True)
    tok = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    template = ChatTemplate(args.model)

    def ids_of(text: str) -> list[int]:
        rendered = template.render([{"role": "user", "content": text}], tools=None, enable_thinking=False)
        return tok.encode(rendered, add_special_tokens=False).ids

    windows = {}
    if args.capture not in ("all", "none", ""):
        for w in args.capture.split(","):
            u, a, b = w.split(":")
            windows[int(u.lstrip("u"))] = (int(a), int(b))
    multi = engine.multi
    if args.capture == "all":
        CAP.start()
    for users in [int(u) for u in args.users.split(",")]:
        for g in range(args.groups):
            nr.forget(engine)
            pick = [(g * users + i) % len(PROMPTS) for i in range(users)]
            streams = [Stream(ids_of(PROMPTS[p]), args.new, None, draft=True, stop_eos=False) for p in pick]
            win = windows.get(users) if g == 0 else None
            torch.cuda.synchronize()
            t_admit = time.perf_counter()
            for s in streams:
                multi.admit(s)
            walls, fills, r = [], [], 0
            while not all(s.done for s in streams):
                decoding = all(len(s.out) >= 1 for s in streams)
                cap = bool(win) and decoding and win[0] <= r < win[1]
                if cap:
                    CAP.start()
                torch.cuda.nvtx.range_push(f"{'round' if decoding else 'fill'} u{users}")
                t = time.perf_counter()
                done = multi.round()
                multi.finish(done)
                dt = time.perf_counter() - t
                torch.cuda.nvtx.range_pop()
                if decoding:
                    walls.append(dt)
                    r += 1
                    if cap and r >= win[1]:
                        CAP.stop()
                else:
                    fills.append(dt)
            if args.capture != "all":
                CAP.stop()
            outs = [len(s.out) for s in streams]
            drafted = sum(s.drafted for s in streams)
            accepted = sum(s.accepted for s in streams)
            dec_tokens = sum(max(0, o - 1) for o in outs)
            dec_s = sum(walls)
            row = {"users": users, "group": g, "prompts": pick, "prompt_tokens": [len(s.prompt) for s in streams],
                   "out": outs, "errors": [repr(s.error) for s in streams if s.error is not None],
                   "fill_s": sum(fills), "rounds": len(walls),
                   "decode_tokens": dec_tokens, "decode_s": dec_s,
                   "agg_tok_s": dec_tokens / dec_s if dec_s else 0.0,
                   "per_user_tok_s": dec_tokens / dec_s / users if dec_s else 0.0,
                   "tokens_per_round_per_user": dec_tokens / len(walls) / users if walls else 0.0,
                   "stream_rounds": sum(s.rounds for s in streams), "drafted": drafted, "accepted": accepted,
                   "acceptance": accepted / drafted if drafted else 0.0, "round": stats_ms(walls),
                   "wall_s": time.perf_counter() - t_admit, "captured": bool(win) or args.capture == "all"}
            report["groups"].append(row)
            print(f"[drive] u{users} g{g}: {row['agg_tok_s']:.1f} tok/s aggregate "
                  f"({row['per_user_tok_s']:.1f} a user), {row['tokens_per_round_per_user']:.2f} tokens/round/user, "
                  f"acceptance {row['acceptance']:.3f}, "
                  f"round p50 {row['round'].get('p50_ms', 0):.1f} ms, out {outs}", flush=True)
    CAP.stop()
    report["capture_windows"] = CAP.windows
    return report


def ceilings(args) -> dict:
    """Achievable rates in this container, without the engine: CUDA-event medians of repeated calls."""

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False   # fp32 accumulation, as gated
    report: dict = {"mode": "ceilings", "device": torch.cuda.get_device_name(), "sms":
                    torch.cuda.get_device_properties(0).multi_processor_count}

    def timed(fn, reps: int) -> float:
        for _ in range(3):
            fn()
        times = []
        for _ in range(reps):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            b.synchronize()
            times.append(a.elapsed_time(b) / 1e3)
        return statistics.median(times)

    for m, n, k in ((8192, 8192, 8192), (16384, 2560, 1280), (16384, 1280, 2560), (2048, 2560, 8192)):
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).t()
        t = timed(lambda: a @ b, 20)
        report[f"bf16_mm_{m}x{n}x{k}_tflops"] = 2 * m * n * k / t / 1e12
        del a, b
    x = torch.ones(1 << 31, device="cuda", dtype=torch.uint8)
    y = torch.empty_like(x)
    report["copy_gbps"] = 2 * x.numel() / timed(lambda: y.copy_(x), 10) / 1e9      # read + write
    xf = x.view(torch.float32)                         # a float sum streams; an int64 sum is compute-bound (rf-s1)
    report["read_gbps"] = x.numel() / timed(lambda: xf.sum(), 10) / 1e9
    report["write_gbps"] = x.numel() / timed(lambda: y.fill_(3), 10) / 1e9
    print("[drive] ceilings " + ", ".join(f"{k} {v:.1f}" for k, v in report.items() if isinstance(v, float)),
          flush=True)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("prefill", "decode", "ceilings"))
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--tokens", default="8192,32768")
    ap.add_argument("--warm", type=int, default=8192)
    ap.add_argument("--capture", default="all")
    ap.add_argument("--inventory", action="store_true")
    ap.add_argument("--users", default="1,4")
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--kv-dtype", default="int8")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"prefill": prefill, "decode": decode, "ceilings": ceilings}[args.mode](args)
    (args.out / "drive.json").write_text(json.dumps(report, indent=1))
    print(f"[drive] wrote {args.out / 'drive.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
