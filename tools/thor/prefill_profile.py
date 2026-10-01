"""Flash Next's cold prefill on CUDA, split by kernel class (FR-OPS-17 K-PROF): wall time, CUDA events and CUPTI.

    python tools/thor/prefill_profile.py MODEL_DIR OUT_DIR [--tokens 8192,32768,131072] [--parallel 4]
        [--kv-dtype int8] [--modes plain,events,profile] [--profile-every 8] [--gemm-bench] [--save-gdn PATH]
        [--natural 32768] [--rows-sweep 4096,8192 --sweep-tokens 32768] | [--ple-probe 32768]

The engine is built as ``tensorfold serve MODEL_DIR --backend cuda --kv-dtype int8 --parallel 4`` builds it, and each
prompt takes the server's path for one request (``MultiDecoder.admit``, then rounds until its first token), so it fills
in passes of PREFILL_ROWS rows exactly as served. Every prompt is new text (cold: nothing resumes). The kernels are
untouched: the tool wraps the forward's Python steps in labels at run time, and only in the modes that need them.

  plain    no labels: each prompt's wall time (the speed check against the gate's numbers)
  events   a CUDA event at every label boundary: the GPU timeline split by the innermost label (any idle included)
  profile  torch.profiler (CUPTI) on chosen passes (every pass, or every ``--profile-every``-th past 16 passes): each
           kernel's time under the innermost label that launched it, kernel names, busy vs idle

``--gemm-bench`` then times the dense prompt GEMMs (real weights; real activations caught in the first events prompt)
on TensorFold's kernels and on cuBLASLt (bf16, MXFP8 block-scaled, FP8), each with its error against fp32 (G-A), and
``--save-gdn`` keeps the first DeltaNet layer's chain inputs for ``gdn_prefill_bench.py``. ``--natural`` repeats
lengths on prose and code (natural routing and n-gram rows), ``--rows-sweep`` rebuilds the decoder with longer prompt
chunks (the same prompt first: its head logits must not change), and ``--ple-probe`` measures the n-gram staging
alone (rows and pages a pass touches, the host gather's time, the table's page-cache residency). Summaries:
``kprof_summary.py``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from kprof_classes import klass  # noqa: E402

# ----------------------------------------------------------------------------------------------------- labels

class Labels:
    """The label stack: CUDA events at its boundaries (events), profiler ranges (profile), or nothing (plain)."""

    def __init__(self) -> None:
        self.mode = "plain"
        self.stack: list[str] = []
        self.marks: list[tuple] = []          # (event, label path, pass index)
        self.ranges: list = []
        self.flops: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.bytes: dict[str, float] = defaultdict(float)
        self.passes = -1
        self.ctx: dict = {}
        self.capture: dict | None = None      # label -> [inputs] while the GEMM bench's activations are caught

    def mark(self) -> None:
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self.marks.append((ev, tuple(self.stack), self.passes))

    def push(self, label: str, flops: float = 0.0, nbytes: float = 0.0) -> None:
        self.stack.append(label)
        if self.mode == "plain":
            return
        self.calls[label] += 1
        self.flops[label] += flops
        self.bytes[label] += nbytes
        if self.mode == "events":
            self.mark()
        else:
            rf = torch.autograd.profiler.record_function("kp:" + "/".join(self.stack))
            rf.__enter__()
            self.ranges.append(rf)

    def pop(self) -> None:
        self.stack.pop()
        if self.mode == "events":
            self.mark()
        elif self.mode == "profile":
            self.ranges.pop().__exit__(None, None, None)

    def reset(self, mode: str) -> None:
        self.mode, self.marks, self.passes = mode, [], -1
        self.flops, self.calls, self.bytes = defaultdict(float), defaultdict(int), defaultdict(float)


L = Labels()


def wrap(owner, attr: str, meta) -> None:
    """``owner.attr`` as a labelled call; ``meta(*args, **kw)`` gives (label, flops, bytes), a label or None."""

    orig = getattr(owner, attr)

    def labelled(*args, **kwargs):
        if L.mode == "plain":
            return orig(*args, **kwargs)
        got = meta(*args, **kwargs) if callable(meta) else meta
        if got is None:
            return orig(*args, **kwargs)
        label, flops, nbytes = (got, 0.0, 0.0) if isinstance(got, str) else got
        L.push(label, flops, nbytes)
        try:
            return orig(*args, **kwargs)
        finally:
            L.pop()

    labelled.__wrapped__ = orig
    setattr(owner, attr, labelled)


def weight_names(w) -> dict[int, str]:
    """id(linear) -> its name, for every dense linear the forward can hand ``_mm`` or ``bf16.matmul``."""

    names: dict[int, str] = {}

    def put(obj, name: str) -> None:
        if obj is not None:
            names[id(obj)] = name
            inner = getattr(obj, "b", None)          # a bf16 face's matrix (``_readout_b16`` passes hc.down.b)
            if inner is not None:
                names[id(inner)] = name

    def hc(h) -> None:
        if h is None:
            return
        for field, name in (("down", "hc_down"), ("prefill_down", "hc_down"), ("up", "hc_up"),
                            ("prefill_up", "hc_up")):
            put(getattr(h, field, None), name)

    layers = list(w.layers) + ([w.mtp.layer] if w.mtp is not None else [])
    for layer in layers:
        hc(layer.attn_hc)
        hc(layer.mlp_hc)
        if layer.attn is not None:
            put(layer.attn.proj, "attn_qkv")
            put(layer.attn.o, "attn_o")
        if layer.gdn is not None:
            put(layer.gdn.proj, "gdn_in")
            put(layer.gdn.out, "gdn_out")
        if layer.ple is not None:
            put(layer.ple.key, "ple_key")
            put(layer.ple.value, "ple_value")
        shared = getattr(getattr(layer.moe, "experts", None), "shared", None)
        if shared is not None:
            put(shared.gu, "shared_gu")
            put(shared.down, "shared_down")
    hc(w.mixer)
    if w.mtp is not None:
        hc(w.mtp.mixer)
        put(w.mtp.fc_e, "mtp_fc_e")
        put(w.mtp.fc_h, "mtp_fc_h")
    put(w.head, "lm_head")
    put(w.draft_head, "lm_head_draft")
    return names


def instrument(w, cfg) -> None:
    """Wrap every Python step of a prompt pass in a label (the kernels themselves are untouched)."""

    from tensorfold.cuda import moe as moe_mod
    from tensorfold.cuda.kernels import gdn as shared_gdn
    from tensorfold.cuda.nvfp4 import experts as nvx
    from tensorfold.cuda.nvfp4 import thor_experts as thor
    from tensorfold.families.qwen4_exp.cuda import attention as attn_mod
    from tensorfold.families.qwen4_exp.cuda import bf16, forward, glue, gdn_io, mtp, multi, nvfp4_moe

    names = weight_names(w)
    D, NI = cfg.hidden, cfg.moe_width
    budget, ratio = cfg.index_budget, cfg.index_ratio
    dense_keys = budget + ratio - 1

    def catch(name: str, q, x) -> None:
        """The GEMM bench's rows: each name's first linear, its first two calls (two passes)."""

        got = L.capture.setdefault(name, [])
        if not got or (got[0][0] is q and len(got) < 2):
            got.append((q, x.detach().clone()))

    def mm_meta(x, q, *a, **kw):
        name = names.get(id(q), "other")
        n, k = int(getattr(q, "n", 0)), int(getattr(q, "k", 0))
        if L.capture is not None:
            catch(name, q, x)
        return "dense." + name, 2.0 * x.shape[0] * n * k, 0.0

    def b16_meta(x, b, *a, **kw):
        if L.stack and L.stack[-1].startswith("dense."):
            return None                                   # inside ``_mm``: already labelled
        name = names.get(id(b), "other")
        if L.capture is not None:
            catch(name, b, x)
        return "dense." + name, 2.0 * x.shape[0] * b.n * b.k, 0.0

    wrap(forward, "_mm", mm_meta)
    wrap(mtp, "_mm", mm_meta)
    wrap(bf16, "matmul", b16_meta)
    try:                                              # Jetson Thor's prompt GEMMs (THOR-K2), where present
        from tensorfold.cuda.nvfp4 import thor_dense
    except ImportError:
        thor_dense = None
    if thor_dense is not None and hasattr(thor_dense, "linear"):
        def thor_meta(x, lin, out=None):
            if L.stack and L.stack[-1].startswith(("dense.", "moe.shared")):
                return None                               # inside ``_mm`` or the shared expert: labelled there
            return b16_meta(x, lin)

        wrap(thor_dense, "linear", thor_meta)
    for owner in (forward, mtp):
        wrap(owner, "_embed", "embed")
        wrap(owner, "finish", "finish")
    wrap(forward, "ple_block", "ple")
    first_linear = next(layer.index for layer in w.layers if layer.linear)

    def gdn_meta(layer, *a, **kw):
        L.ctx["gdn_layer"] = layer.index
        return "gdn"

    wrap(forward, "gdn_block", gdn_meta)
    wrap(forward, "attn_block", "attn")
    wrap(forward, "moe_block", "moe")
    wrap(forward, "shift_windows", "shift")
    for name, label in (("hc_writeback", "hc.writeback"), ("hc_normed", "hc.normed"), ("hc_act", "hc.act"),
                        ("hc_reduce_act", "hc.act"), ("hc_mix", "hc.mix"), ("attn_prep", "attn.prep"),
                        ("attn_gate", "attn.gate"), ("ple_embed", "ple.embed"), ("ple_embed_bf16", "ple.embed"),
                        ("ple_gate", "ple.gate"), ("ple_conv", "ple.conv"), ("rmsnorm", "norm.rms"),
                        ("add_streams", "elt.add_streams")):
        wrap(glue, name, label)

    def chain_meta(q, k, v, g, beta, state, final):
        rows, hv = v.shape[0], v.shape[1]
        if L.capture is not None and L.ctx.get("gdn_layer") == first_linear and "mtp" not in L.stack:
            got = L.capture.setdefault("gdn_chain", [])     # the first DeltaNet layer's first two passes, as run
            if len(got) < 2:
                got.append((first_linear, {"q": q.clone(), "k": k.clone(), "v": v.clone(), "g": g.clone(),
                                           "beta": beta.clone(), "state": state.clone()}))
        return "gdn.chain", 7.0 * rows * hv * cfg.dk * cfg.dv, 0.0

    wrap(gdn_io, "front", "gdn.front")
    wrap(gdn_io, "back", "gdn.back")
    wrap(shared_gdn, "chain", chain_meta)

    def core_meta(q, kc, vc, pos0, scratch, rows, scale, out=None, *, context=None, **kw):
        if context is None:
            return "attn.core"
        p0 = int(context) - int(rows)
        keys = sum(min(p0 + i + 1, dense_keys) for i in range(int(rows)))
        return "attn.core", 4.0 * q.shape[1] * q.shape[2] * keys, 0.0

    def rows_meta(iq, pooled, pos0, scratch, rows, *, context=None):
        blocks = 0 if context is None else -(-int(context) // ratio)
        return "attn.qsa_rows", 2.0 * int(rows) * iq.shape[1] * pooled.shape[1] * blocks, 0.0

    wrap(attn_mod, "attention", core_meta)
    wrap(attn_mod, "qsa_pool", "attn.qsa_pool")
    wrap(attn_mod, "qsa_rows", rows_meta)

    def router_meta(x, rows, out=None):
        return "moe.router", 2.0 * x.shape[0] * rows.shape[0] * x.shape[1], 0.0

    wrap(moe_mod, "router", router_meta)
    wrap(moe_mod, "select", "moe.select")

    def run_meta(x, ex, plan, y, rows, top_k, scr):
        L.ctx["pairs"] = int(rows) * int(top_k)
        L.ctx["wbytes"] = (ex.w1.numel() + ex.w1_sf.numel(), ex.w2.numel() + ex.w2_sf.numel())
        return "moe.routed"

    def gemm_meta(fw, w_, sf, alpha, act_in, out, tiles, scr, experts, gated, t):
        pairs = L.ctx.get("pairs", 0)
        if gated:
            return "moe.routed.gemm1", 2.0 * pairs * 2 * NI * D, float(L.ctx.get("wbytes", (0, 0))[0])
        return "moe.routed.gemm2", 2.0 * pairs * D * NI, float(L.ctx.get("wbytes", (0, 0))[1])

    wrap(thor, "run", run_meta)
    wrap(thor, "_gemm", gemm_meta)

    def portable_meta(which):
        def meta(x, ex, plan, out, rows, **kw):
            pairs = int(rows) * (plan.slots - 1)
            return f"moe.portable.{which}", 2.0 * pairs * (2 * NI * D if which == "gate_up" else D * NI), 0.0
        return meta

    wrap(nvx, "gate_up", portable_meta("gate_up"))
    wrap(nvx, "down", portable_meta("down"))

    def shared_meta(lin, x, prompt, f32):
        if L.capture is not None:
            catch(names.get(id(lin), "shared"), lin, x)
        return "moe.shared.gemm", 2.0 * x.shape[0] * int(lin.n) * int(lin.k), 0.0

    wrap(nvfp4_moe, "_shared", shared_meta)
    wrap(nvfp4_moe.MoE4, "shared_act", "moe.shared.act")
    for name, label in (("stage", "stage"), ("compute", "compute"), ("commit", "commit"),
                        ("cut_snapshot", "snapshot"), ("mtp_compute", "mtp"), ("mtp_stage", "mtp.stage")):
        wrap(multi, name, label)

    def pass_meta(self):
        L.passes += 1
        return "pass"

    md = multi.MultiDecoder
    wrap(md, "_pass", pass_meta)
    wrap(md, "admit", "admit")
    wrap(md, "_absorb", "absorb")
    wrap(md, "_joined", "joined")
    wrap(md, "_make_room", "make_room")


# ------------------------------------------------------------------------------------------------ prompts

WORDS = ["alpha", "bravo", "cobalt", "delta", "ember", "fjord", "garnet", "harbor", "indigo", "juniper", "kelp",
         "lumen", "meadow", "nectar", "onyx", "pylon", "quartz", "raven", "sierra", "tundra"]


def natural_text(seed: int, chars: int) -> str:
    """Prose and code from this tree (docs, then sources), from a seeded offset: a prompt with natural routing."""

    root = Path(__file__).resolve().parents[2]
    files = sorted(root.glob("docs/**/*.md")) + sorted(root.glob("*.md")) + sorted(root.glob("src/**/*.py"))
    text = "\n\n".join(f.read_text(errors="ignore") for f in files)
    start = random.Random(seed).randrange(max(1, len(text) - chars))
    return (text + text)[start:start + chars]


def prompt_ids(model_dir: Path, tokens: int, seed: int, natural: bool = False) -> list[int]:
    """``tokens`` ids of a new user message in the server's template, thinking off: prefill_ttft.py's words and
    numbers, or (``natural``) prose and code from this tree."""

    from tokenizers import Tokenizer

    from tensorfold.cuda.chat_template import ChatTemplate

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    rng = random.Random(seed)
    if natural:
        words = [f"Run {seed}. Summarize the following.", natural_text(seed, tokens * 6)]
    else:
        words = [f"Run {seed}."] + [WORDS[rng.randrange(len(WORDS))] if i % 3 else str(rng.randrange(100000))
                                     for i in range(tokens * 40 // 100)]
    marker = "⁣MARK⁣"
    text = ChatTemplate(model_dir).render([{"role": "user", "content": marker}], tools=None, enable_thinking=False)
    head, tail = text.split(marker)
    h = tok.encode(head, add_special_tokens=False).ids
    t = tok.encode("\n\nReply with one word." + tail, add_special_tokens=False).ids
    body = tok.encode(" ".join(words), add_special_tokens=False).ids
    room = tokens - len(h) - len(t)
    if len(body) < room:
        raise ValueError(f"the text gave {len(body)} tokens, {room} needed")
    return h + body[:room] + t


# ------------------------------------------------------------------------------------------------ one prompt

def run_prompt(engine, ids: list[int]) -> dict:
    """The server's path for one request with one token out; returns wall seconds and the first token."""

    from tensorfold.cuda.streams import Stream

    import hashlib

    from tensorfold.families.qwen4_exp.cuda import multi as multi_mod

    multi = engine.multi
    md = multi_mod.MultiDecoder
    joined = md._joined
    seen: list = []

    def hashed(dec, pieces, heads, lasts, spent):         # the prompt's head logits, bit for bit (after the wall)
        if heads is not None:
            seen.append(heads.detach().clone())
        return joined(dec, pieces, heads, lasts, spent)

    md._joined = hashed
    s = Stream(list(ids), 1, None, draft=True)
    torch.cuda.synchronize()
    if L.mode == "events":
        L.mark()
    t0 = time.perf_counter()
    try:
        multi.admit(s)
        while not s.done:
            done = multi.round()
            multi.finish(done)
        t1 = time.perf_counter()
    finally:
        md._joined = joined
    if L.mode == "events":
        L.mark()
    torch.cuda.synchronize()
    digest = hashlib.sha256(seen[-1].view(torch.int16).cpu().numpy().tobytes()).hexdigest()[:16] if seen else None
    return {"tokens": len(ids), "seconds": t1 - t0, "tok_s": len(ids) / (t1 - t0), "first": s.out[:1],
            "logits_sha": digest, "error": None if s.error is None else repr(s.error)}


def events_summary(marks: list) -> dict:
    """The GPU timeline between consecutive marks, by innermost label, class and pass."""

    torch.cuda.synchronize()
    by_path: dict[str, float] = defaultdict(float)
    by_class: dict[str, float] = defaultdict(float)
    by_pass: dict[int, dict] = defaultdict(lambda: defaultdict(float))
    total = 0.0
    for (a, path, p), (b, _, _) in zip(marks, marks[1:]):
        ms = a.elapsed_time(b)
        total += ms
        by_path["/".join(path) or "(top)"] += ms
        c = klass(path)
        by_class[c] += ms
        by_pass[p][c] += ms
    return {"total_ms": total, "classes": dict(by_class), "paths": dict(by_path),
            "passes": {str(k): dict(v) for k, v in sorted(by_pass.items())}}


# ------------------------------------------------------------------------------------------------ CUPTI

def innermost(annots: list, queries: list) -> dict:
    """annots [(start, end, path)] properly nested on one thread; queries [(ts, key)] -> {key: innermost path}."""

    annots = sorted(annots, key=lambda a: (a[0], -a[1]))
    out, stack, i = {}, [], 0
    for ts, key in sorted(queries):
        while i < len(annots) and annots[i][0] <= ts:
            while stack and stack[-1][1] < annots[i][0]:
                stack.pop()
            stack.append(annots[i])
            i += 1
        while stack and stack[-1][1] < ts:
            stack.pop()
        out[key] = stack[-1][2] if stack else None
    return out


def trace_summary(path: Path) -> dict:
    """A pass's Chrome trace -> GPU time by label path and class, kernel names, busy and span."""

    data = json.loads(path.read_text())
    events = data["traceEvents"] if isinstance(data, dict) else data
    annots: dict = defaultdict(list)
    launches: dict = {}
    gpu: list = []
    for e in events:
        cat = e.get("cat")
        if cat == "user_annotation" and str(e.get("name", "")).startswith("kp:"):
            annots[e.get("tid")].append((float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0)),
                                         tuple(e["name"][3:].split("/"))))
        elif cat in ("cuda_runtime", "cuda_driver"):
            corr = (e.get("args") or {}).get("correlation")
            if corr is not None:
                launches[corr] = (float(e["ts"]), e.get("tid"))
        elif cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            gpu.append(e)
    queries: dict = defaultdict(list)
    for i, e in enumerate(gpu):
        launch = launches.get((e.get("args") or {}).get("correlation"))
        if launch is not None:
            queries[launch[1]].append((launch[0], i))
    paths: dict = {}
    for tid, qs in queries.items():
        paths.update(innermost(annots.get(tid, []), qs))
    by_path: dict = defaultdict(float)
    by_class: dict = defaultdict(float)
    count: dict = defaultdict(int)
    kernels: dict = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    spans = []
    unattributed = 0.0
    for i, e in enumerate(gpu):
        dur = float(e.get("dur", 0)) / 1000.0
        p = paths.get(i)
        if p is None:
            unattributed += dur
            p = ("(unattributed)",)
        key = "/".join(p)
        by_path[key] += dur
        count[key] += 1
        c = klass(p) if p != ("(unattributed)",) else "unattributed"
        by_class[c] += dur
        name = e.get("name", "?")
        name = name if len(name) <= 140 else name[:140] + "..."
        k = kernels[c][name]
        k[0] += dur
        k[1] += 1
        spans.append((float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0))))
    spans.sort()
    busy, end = 0.0, None
    for a, b in spans:
        if end is None or a > end:
            busy += b - a
            end = b
        elif b > end:
            busy += b - end
            end = b
    span = (spans[-1][1] - spans[0][0]) / 1000.0 if spans else 0.0
    top = {c: sorted(([n, round(v[0], 3), v[1]] for n, v in ks.items()), key=lambda r: -r[1])[:12]
           for c, ks in kernels.items()}
    return {"busy_ms": busy / 1000.0, "span_ms": span, "kernel_ms": sum(by_class.values()),
            "unattributed_ms": unattributed, "gpu_events": len(gpu), "classes": dict(by_class),
            "paths": dict(by_path), "counts": dict(count), "kernels": top}


class PassProfiler:
    """torch.profiler around chosen prompt passes (drained before and after, so a trace holds one pass alone)."""

    def __init__(self, out: Path, tag: str, chosen, last: int) -> None:
        self.out, self.tag, self.chosen, self.last = out, tag, chosen, last
        self.summaries: dict[int, dict] = {}
        self.prof = None
        self.index = -1

    def install(self) -> None:
        from tensorfold.families.qwen4_exp.cuda import multi

        md = multi.MultiDecoder
        labelled = md._pass
        prof_self = self

        def profiled(dec):
            prof_self.index += 1
            i = prof_self.index
            if not prof_self.chosen(i):
                return labelled(dec)
            torch.cuda.synchronize()
            p = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA])
            p.__enter__()
            try:
                return labelled(dec)
            finally:
                torch.cuda.synchronize()
                p.__exit__(None, None, None)
                path = prof_self.out / f"trace_{prof_self.tag}_pass{i:03d}.json"
                p.export_chrome_trace(str(path))
                prof_self.summaries[i] = trace_summary(path)
                if prof_self.keep(i):                     # the raw trace of the first and last pass, gzipped
                    with open(path, "rb") as src, gzip.open(str(path) + ".gz", "wb") as dst:
                        dst.write(src.read())
                path.unlink()

        profiled.__wrapped__ = labelled
        self.installed = (md, labelled)
        md._pass = profiled

    def keep(self, i: int) -> bool:
        return i == 0 or i == self.last

    def remove(self) -> None:
        md, labelled = self.installed
        md._pass = labelled


# ------------------------------------------------------------------------------------------------ GEMM bench

def _timed(fn, iters: int) -> float:
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def to_blocked(scales: torch.Tensor) -> torch.Tensor:
    """e8m0 scales [rows, K/32] -> cuBLASLt's 128 x 4 block layout (rows and columns padded), flattened."""

    rows, cols = scales.shape
    r, c = -(-rows // 128) * 128, -(-cols // 4) * 4
    padded = torch.full((r, c), 127, dtype=torch.uint8, device=scales.device)
    padded[:rows, :cols] = scales.view(torch.uint8)
    blocks = padded.view(r // 128, 128, c // 4, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1).contiguous().view(torch.float8_e8m0fnu)


def mx_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [M, K] -> MXFP8: e4m3 values and an e8m0 power of two per 32 inputs, amax / 448 rounded up (no clipping)."""

    m, k = x.shape
    g = x.float().view(m, k // 32, 32)
    amax = g.abs().amax(dim=-1).clamp(min=2.0 ** -120)
    e = torch.ceil(torch.log2(amax / 448.0))
    q = (g / torch.exp2(e)[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(m, k)
    return q, (e + 127.0).clamp(0, 254).to(torch.uint8)


def dense_bench(captured: dict, out: Path) -> dict:
    """Each captured dense linear (real weights, real rows): TensorFold's kernel vs cuBLASLt, time and G-A error."""

    from tensorfold.cuda.nvfp4.linear import Mx8Linear
    from tensorfold.families.qwen4_exp.cuda import bf16 as b16mod
    from tensorfold.families.qwen4_exp.cuda import nvfp4 as fam_nvfp4

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    t_end = time.time() + 3.0                         # Jetson clocks ramp: a few seconds of GEMMs first
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    while time.time() < t_end:
        a @ a
    torch.cuda.synchronize()
    results = {}
    for name, items in sorted(captured.items()):
        lin, x0 = items[0]
        rows = [x for q, x in items if q is lin]
        x = torch.cat(rows)[:4096].contiguous() if len(rows) > 1 else rows[0]
        w8 = sc = None
        if isinstance(lin, Mx8Linear):
            w8 = lin.w8_rows()
            sc = lin.scale_rows().contiguous()
            wf = w8.float() * torch.exp2(sc.float() - 127.0).repeat_interleave(32, dim=1)
            current = lambda xx: lin.prefill(xx)                                          # noqa: E731
            kind = "mxfp8"
        elif isinstance(lin, fam_nvfp4.FP4):
            wf = fam_nvfp4.dequantize_fp4(lin).float()
            current = lambda xx: fam_nvfp4.matmul(xx, lin)                               # noqa: E731
            kind = "fp4table" if not lin.packed else "nvfp4"
        elif hasattr(lin, "b") or isinstance(lin, b16mod.B16):
            bmat = lin.b if hasattr(lin, "b") else lin
            wf = bmat.weight.float()
            current = lambda xx, bmat=bmat: b16mod.matmul(xx, bmat)                      # noqa: E731
            kind = "bf16"
        else:
            results[name] = {"skipped": type(lin).__name__}
            continue
        n, k = wf.shape
        wb = wf.to(torch.bfloat16)
        exact_bf16 = bool(torch.equal(wb.float(), wf))
        row = {"kind": kind, "n": n, "k": k, "weights_exact_in_bf16": exact_bf16, "by_m": {}}
        for m in (128, 512, 2048, 4096):
            if m > x.shape[0]:
                continue
            xm = x[:m].contiguous()
            ref = xm.float() @ wf.t()
            norm = ref.norm().item()

            def err(y):
                return ((y.float() - ref).norm().item() / norm) if norm > 0 else 0.0

            cands = {"tensorfold": lambda: current(xm), "cublas_bf16": lambda: xm @ wb.t()}
            if kind == "mxfp8":
                xq, xs = mx_quantize(xm)
                sa, sb = to_blocked(xs), to_blocked(sc)
                wq = w8.contiguous()
                cands["cublaslt_mxfp8"] = lambda: torch._scaled_mm(xq, wq.t(), scale_a=sa, scale_b=sb,
                                                                   out_dtype=torch.bfloat16)
                cands["mxfp8_act_quant"] = lambda: mx_quantize(xm)
                if hasattr(lin, "prefill8"):
                    cands["tensorfold_fp8_prompt"] = lambda: lin.prefill8(xm)
            got = {}
            for label, fn in cands.items():
                try:
                    y = fn()
                    torch.cuda.synchronize()
                    e = None if label == "mxfp8_act_quant" else err(y if not isinstance(y, tuple) else y[0])
                    iters = max(3, min(200, int(2e12 / (2.0 * m * n * k + 1))))
                    _timed(fn, 2)
                    ms = statistics.median(_timed(fn, iters) for _ in range(3))
                    got[label] = {"ms": ms, "tflops": 2.0 * m * n * k / ms / 1e9, "rel_err": e}
                except Exception as exc:                          # noqa: BLE001  (a path this torch lacks)
                    got[label] = {"error": repr(exc)[:300]}
            base = got.get("tensorfold", {}).get("rel_err")
            for label, r in got.items():
                if base and r.get("rel_err") is not None:
                    r["err_ratio"] = r["rel_err"] / base
            row["by_m"][str(m)] = got
        results[name] = row
        print(f"[gemm] {name} {kind} [{n} x {k}]: " + "; ".join(
            f"{lab} {r.get('ms', float('nan')):.3f} ms {r.get('tflops', 0):.0f} TF "
            f"err x{r.get('err_ratio', float('nan')):.2f}"
            for lab, r in row["by_m"].get("2048", {}).items() if "ms" in r), flush=True)
    (out / "gemm_bench.json").write_text(json.dumps(results, indent=1))
    return results


# ------------------------------------------------------------------------------------------------ main

def build_engine(model_dir: Path, parallel: int, kv_dtype: str):
    """The engine as ``tensorfold serve MODEL --backend cuda --kv-dtype KV --parallel N`` builds it."""

    from tensorfold.cuda import prompt_precision
    from tensorfold.families import qwen4_exp

    prompt_precision.set_fp8(prompt_precision.FP8_BY_DEFAULT)
    options = {"drafter": "", "tp": 1, "rank": 0, "master": "", "master_port": 29551, "no_drafts": False,
               "kv_dtype": kv_dtype, "context": None, "context_explicit": False}
    if parallel > 1:
        options["parallel"] = parallel
    return qwen4_exp.cuda_engine(model_dir, **options)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--tokens", default="8192,32768,131072")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--kv-dtype", default="int8")
    ap.add_argument("--modes", default="plain,events,profile")
    ap.add_argument("--profile-every", type=int, default=8,
                    help="past 16 passes, profile every Nth pass (and the last)")
    ap.add_argument("--gemm-bench", action="store_true")
    ap.add_argument("--natural", default="", help="lengths also run on natural text (plain and events)")
    ap.add_argument("--rows-sweep", default="", help="prompt chunk rows to rebuild the decoder with, e.g. 4096,8192")
    ap.add_argument("--sweep-tokens", default="32768", help="lengths the chunk sweep runs (plain and events)")
    ap.add_argument("--save-gdn", type=Path, default=None, help="save the first DeltaNet layer's chain inputs")
    ap.add_argument("--ple-probe", default="", help="lengths whose n-gram staging is measured (host only), then exit")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lengths = [int(v) for v in args.tokens.split(",")]
    modes = args.modes.split(",")
    t0 = time.perf_counter()
    engine = build_engine(args.model, args.parallel, args.kv_dtype)
    w = engine.w
    cfg = w.cfg
    print(f"[kprof] engine up in {time.perf_counter() - t0:.0f}s; layers {cfg.layers}, D {cfg.hidden}, experts "
          f"{cfg.experts} top-{cfg.top_k} NI {cfg.moe_width}; prefill rows {engine.multi.prefill_rows}", flush=True)
    if args.ple_probe:
        report = ple_probe(engine, args.model, [int(v) for v in args.ple_probe.split(",")])
        (args.out / "ple_probe.json").write_text(json.dumps(report, indent=1))
        return
    instrument(w, cfg)
    from tensorfold.families.qwen4_exp.cuda import nvfp4_moe

    thor_layers = len(nvfp4_moe.thor_layers(w))
    report = {"model": str(args.model), "parallel": args.parallel, "kv_dtype": args.kv_dtype,
              "prefill_rows": engine.multi.prefill_rows, "thor_layers": thor_layers,
              "torch": torch.__version__, "runs": []}
    seed = 1000
    captured = None
    for mode in modes:
        for n in lengths:
            seed += 1
            ids = prompt_ids(args.model, n, seed)
            L.reset(mode)
            prof = None
            if mode == "profile":
                passes = -(-n // engine.multi.prefill_rows)
                last = passes - 1
                every = 1 if passes <= 16 else args.profile_every
                prof = PassProfiler(args.out, f"{n}", lambda i, e=every, z=last: i % e == 0 or i == z, last)
                prof.install()
            if mode == "events" and captured is None and args.gemm_bench:
                L.capture = {}
            try:
                got = run_prompt(engine, ids)
            finally:
                if prof is not None:
                    prof.remove()
                if L.capture is not None:
                    captured, L.capture = _thin(L.capture), None
            row = {"mode": mode, "target": n, **got, "flops": dict(L.flops), "calls": dict(L.calls),
                   "bytes": dict(L.bytes)}
            if mode == "events":
                row["events"] = events_summary(L.marks)
                L.marks = []
            if prof is not None:
                row["profile"] = {str(k): v for k, v in sorted(prof.summaries.items())}
            report["runs"].append(row)
            print(f"[kprof] {mode} {n}: {got['tokens']} tokens in {got['seconds']:.2f}s = {got['tok_s']:.0f} tok/s, "
                  f"first {got['first']}, logits {got['logits_sha']}"
                  + (f" (events span {row['events']['total_ms'] / 1000:.2f}s)" if mode == "events" else "")
                  + (f" ({len(prof.summaries)} passes profiled)" if prof else ""), flush=True)
            (args.out / "kprof.json").write_text(json.dumps(report, indent=1))
    L.reset("plain")
    if captured and args.save_gdn and captured.get("gdn_chain"):
        torch.save([d for _, d in captured.pop("gdn_chain")], str(args.save_gdn))
        print(f"[kprof] DeltaNet chain inputs saved to {args.save_gdn}", flush=True)
    if args.gemm_bench and captured:
        captured.pop("gdn_chain", None)
        report["gemm_bench"] = dense_bench(captured, args.out)
        (args.out / "kprof.json").write_text(json.dumps(report, indent=1))
    captured = None

    def one(mode: str, n: int, natural: bool, tag: dict) -> None:
        nonlocal seed
        seed += 1
        ids = prompt_ids(args.model, n, seed, natural=natural)
        L.reset(mode)
        got = run_prompt(engine, ids)
        row = {"mode": mode, "target": n, **tag, **got, "flops": dict(L.flops), "calls": dict(L.calls),
               "bytes": dict(L.bytes)}
        if mode == "events":
            row["events"] = events_summary(L.marks)
            L.marks = []
        report["runs"].append(row)
        print(f"[kprof] {mode} {n} {tag}: {got['seconds']:.2f}s = {got['tok_s']:.0f} tok/s, first {got['first']}, "
              f"logits {got['logits_sha']}", flush=True)
        (args.out / "kprof.json").write_text(json.dumps(report, indent=1))
        L.reset("plain")

    for n in [int(v) for v in args.natural.split(",") if v]:
        for mode in ("plain", "events"):
            one(mode, n, True, {"text": "natural"})
    sweep = [int(v) for v in args.rows_sweep.split(",") if v]
    if sweep:
        same = int(time.time()) % 100000             # one prompt for every chunk size: their logits must agree
        for rows in [engine.multi.prefill_rows] + sweep:
            if rows != engine.multi.prefill_rows:
                room = rebuild(engine, rows)
                print(f"[kprof] decoder rebuilt with {rows}-row prompt chunks: {room}", flush=True)
                report.setdefault("rebuilds", []).append({"rows": rows, **room})
            for n in [int(v) for v in args.sweep_tokens.split(",") if v]:
                ids = prompt_ids(args.model, n, 900000 + same)
                L.reset("plain")
                got = run_prompt(engine, ids)
                report["runs"].append({"mode": "plain", "target": n, "rows": rows, "text": "same", **got})
                print(f"[kprof] rows {rows} same-prompt {n}: {got['seconds']:.2f}s = {got['tok_s']:.0f} tok/s, "
                      f"first {got['first']}, logits {got['logits_sha']}", flush=True)
                for mode in ("plain", "events"):
                    one(mode, n, False, {"rows": rows})
    print(f"[kprof] done in {time.perf_counter() - t0:.0f}s", flush=True)


def resident(arrays) -> tuple[int, int]:
    """(resident pages, pages) of memory-mapped arrays, by mincore on their mappings."""

    import ctypes
    import mmap as _mmap

    libc = ctypes.CDLL(None, use_errno=True)
    libc.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
    page = _mmap.PAGESIZE
    have = total = 0
    for arr in arrays:
        at = arr.ctypes.data
        start = at - at % page
        size = arr.nbytes + (at - start)
        n = -(-size // page)
        vec = (ctypes.c_ubyte * n)()
        if libc.mincore(ctypes.c_void_p(start), ctypes.c_size_t(size), vec) == 0:
            have += sum(b & 1 for b in vec)
        total += n
    return have, total


def ple_probe(engine, model: Path, lengths: list[int]) -> dict:
    """The n-gram staging alone (host): rows and pages a pass touches, the gather time, page-cache residency."""

    import numpy as np

    w = engine.w
    layer = next(layer for layer in w.layers if layer.ple is not None)
    table, ngram = layer.ple.table, layer.ple.ngram
    arrays = list(getattr(table, "values", [])) + list(getattr(table, "scales", []))
    rows_bytes = int(table.nbytes) // max(1, int(table.rows))
    out = {"table": type(table).__name__, "table_gib": table.nbytes / 2**30, "rows": int(table.rows),
           "row_bytes": rows_bytes, "heads": int(ngram.heads), "runs": []}
    have, total = resident(arrays)
    out["resident_before"] = have / max(1, total)
    print(f"[ple] {out['table']} {out['table_gib']:.2f} GiB, {out['rows']} rows of {rows_bytes} B, "
          f"{ngram.heads} heads; resident {100 * have / max(1, total):.1f}%", flush=True)
    rows = engine.multi.prefill_rows
    for natural in (False, True):
        for n in lengths:
            ids = prompt_ids(model, n, 7000 + n + natural, natural=natural)
            history = np.zeros((0,), dtype=np.int64)
            for repeat in (0, 1):
                hist = history
                per = []
                t_all = time.perf_counter()
                for a in range(0, len(ids), rows):
                    toks = np.asarray(ids[a:a + rows], dtype=np.int64)
                    rid = ngram.ids(hist, toks)
                    hist = np.concatenate([hist, toks])[-(ngram.n - 1):]
                    flat = np.asarray(rid, dtype=np.int64).reshape(-1)
                    pages = np.unique((flat * rows_bytes) // 4096).size
                    t = time.perf_counter()
                    table.gather(rid)
                    per.append({"lookups": int(flat.size), "rows": int(np.unique(flat).size), "pages": int(pages),
                                "ms": 1000 * (time.perf_counter() - t)})
                run = {"tokens": n, "natural": natural, "repeat": repeat, "ms": 1000 * (time.perf_counter() - t_all),
                       "passes": len(per), "lookups": sum(p["lookups"] for p in per),
                       "distinct_rows_per_pass": sum(p["rows"] for p in per) / len(per),
                       "pages_per_pass": sum(p["pages"] for p in per) / len(per),
                       "gather_ms_per_pass": sum(p["ms"] for p in per) / len(per),
                       "first_pass_ms": per[0]["ms"]}
                have, total = resident(arrays)
                run["resident_after"] = have / max(1, total)
                out["runs"].append(run)
                print(f"[ple] {'natural' if natural else 'synthetic'} {n} {'again' if repeat else 'first'}: "
                      f"{run['ms']:.0f} ms, {run['gather_ms_per_pass']:.0f} ms a pass, "
                      f"{run['distinct_rows_per_pass']:.0f} rows and {run['pages_per_pass']:.0f} pages a pass; "
                      f"resident {100 * run['resident_after']:.1f}%", flush=True)
    return out


def rebuild(engine, rows: int) -> dict:
    """The concurrent decoder again with ``rows``-row prompt chunks (the same weights; the old one's memory freed)."""

    import gc

    from tensorfold.families.qwen4_exp.cuda.engine import KEEP
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    old = engine.multi
    slots = len(old.free) + len(old.streams) + len(old.filling) + len(old.kept)
    if getattr(engine, "scheduler", None) is not None:
        engine.scheduler.decoder = None
    engine.multi = None
    del old
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    t = time.perf_counter()
    md = MultiDecoder(engine.w, slots=slots, capacity=engine.max_len, depth=engine.depth,
                      confidence=engine.confidence, keep=KEEP, kv_dtype=engine.kv_dtype, share=0.0, prefill_rows=rows)
    built = time.perf_counter() - t
    md.warm()
    engine.multi = md
    if getattr(engine, "scheduler", None) is not None:
        engine.scheduler.decoder = md
    pbuf = sum(t.untyped_storage().nbytes() for t in _tensors_of(md.pbuf))
    return {"slots": slots, "allocated_gib": (torch.cuda.memory_allocated() - before) / 2**30,
            "pbuf_gib": pbuf / 2**30, "gate_room_gib": md.memory_gate.room / 2**30,
            "build_s": built, "warm_s": time.perf_counter() - t - built}


def _tensors_of(obj, seen=None, depth=0) -> list:
    """The CUDA tensors an object holds (attributes, lists, dicts), each storage once."""

    seen = set() if seen is None else seen
    if depth > 5 or id(obj) in seen:
        return []
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        key = ("s", obj.untyped_storage().data_ptr())
        if not obj.is_cuda or key in seen:
            return []
        seen.add(key)
        return [obj]
    items = obj.values() if isinstance(obj, dict) else obj if isinstance(obj, (list, tuple)) else \
        getattr(obj, "__dict__", {}).values()
    if not isinstance(obj, (dict, list, tuple)) and type(obj).__module__.startswith(("torch", "builtins", "numpy")):
        return []
    out = []
    for v in items:
        out += _tensors_of(v, seen, depth + 1)
    return out


def _thin(capture: dict) -> dict:
    """Keep each linear name's first layer: its first two passes' rows (4096 at most)."""

    out = {}
    for name, items in capture.items():
        first = items[0][0]
        out[name] = items[:2] if name == "gdn_chain" else [(q, x) for q, x in items if q is first][:2]
    return out


if __name__ == "__main__":
    sys.exit(main())
