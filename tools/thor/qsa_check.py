"""QSA prompt attention on Jetson Thor (THOR-K4-QSA): what a prompt's sparse rows select, and what that means for a
kernel that shares key loads across rows.

    python tools/thor/qsa_check.py select MODEL_DIR OUT_DIR [--prompts syn32k,nat32k,syn128k]

``select``: the engine as served (int8 KV, --parallel 4, the Thor switches as they are in this process) fills each
prompt from row 0; after every prompt attention block's ``qsa_rows`` (256 rows, one attention layer) the block's
selections are read back (``AttnScratch.ids``: each sparse row's top blocks of ``ratio`` keys, in block order) and
summarised on the GPU:

- union: the distinct blocks that T consecutive rows (an absolute T-row grid) select, at a grain of g blocks (g = 1:
  the 4-key blocks themselves; g = 4 and 16: 16- and 64-key tiles), for T in 1, 4, 16, 64. A kernel that loads each
  key tile once for T rows and masks the rest computes T x union x g x ratio keys a tile where the rows need T x top x
  ratio: union x g / top is its work inflation;
- locality: the share of a row's blocks among its last 64 and 512 complete blocks, and among the first 64;
- overlap: the share of a row's blocks that the row before it also selects;
- margins: the gap between a row's top-th and next block score, relative to the top-th (how often a score rounding of
  that size could change a selection).

Writes qsa_select.json (per prompt, attention layer and 8k position bucket) and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import ngram_residency as nr  # noqa: E402
import prefill_profile as pp  # noqa: E402

TILES = (1, 4, 16, 64)
GRAINS = (1, 4, 16)
BUCKET = 8192


class Selections:
    """``qsa_rows`` wrapped: each call's selections summarised into per-(prompt, layer, bucket) sums on the GPU."""

    def __init__(self, attn_mod, forward) -> None:
        self.prompt = ""
        self.layer = None
        self.sums: dict = defaultdict(lambda: defaultdict(float))
        self.calls = 0
        self.seconds = 0.0
        orig_rows, orig_block = attn_mod.qsa_rows, forward.attn_block

        def attn_block(layer, w, segs, b, R, mtp, context=None):
            self.layer = "mtp" if mtp else int(layer.index)
            return orig_block(layer, w, segs, b, R, mtp, context=context)

        def qsa_rows(iq, pooled, pos0, scratch, rows, *, context=None):
            orig_rows(iq, pooled, pos0, scratch, rows, context=context)
            if context is not None and self.prompt:
                t = time.perf_counter()
                self.summarise(scratch, int(context) - int(rows), int(rows))
                self.seconds += time.perf_counter() - t

        forward.attn_block, attn_mod.qsa_rows = attn_block, qsa_rows

    @torch.no_grad()
    def summarise(self, scratch, p0: int, n: int) -> None:
        sp = scratch.sparse[:n] != 0
        ratio, top = scratch.ratio, scratch.budget // scratch.ratio
        idx = torch.nonzero(sp).squeeze(1)
        if idx.numel() == 0:
            return
        self.calls += 1
        dev = idx.device
        blocks = (scratch.ids[:n, 0:top * ratio:ratio][idx] // ratio).long()          # [ns, top] in block order
        pos = p0 + idx                                                                 # each sparse row's position
        complete = (pos + 1) // ratio
        nb = int(p0 + n) // ratio + 1
        key = (self.prompt, str(self.layer), (p0 // BUCKET) * BUCKET)
        s = self.sums[key]
        s["rows"] += idx.numel()
        rec64 = (blocks >= (complete - 64)[:, None]).sum()
        rec512 = (blocks >= (complete - 512)[:, None]).sum()
        first = (blocks < 64).sum()
        s["recent64"] += float(rec64)
        s["recent512"] += float(rec512)
        s["first64"] += float(first)
        # per-row occupancy at grain 1: the next row's overlap with this one
        occ1 = torch.zeros((idx.numel(), nb), dtype=torch.bool, device=dev)
        occ1.scatter_(1, blocks, True)
        adjacent = (idx[1:] - idx[:-1]) == 1
        if adjacent.any():
            both = (occ1[1:] & occ1[:-1]).sum(1).float()[adjacent]
            s["overlap_sum"] += float(both.sum()) / top
            s["overlap_n"] += float(adjacent.sum())
        for g in GRAINS:
            bg = blocks // g
            width = nb // g + 1
            for T in TILES:
                tile = pos // T
                uniq, inv, counts = torch.unique(tile, return_inverse=True, return_counts=True)
                occ = torch.zeros((uniq.numel(), width), dtype=torch.bool, device=dev)
                occ[inv[:, None].expand_as(bg), bg] = True
                union = occ.sum(1).float()
                full = counts == T
                if full.any():
                    u = union[full] * g / top                                         # inflation per full tile
                    s[f"infl_g{g}_T{T}_sum"] += float(u.sum())
                    s[f"infl_g{g}_T{T}_sq"] += float((u * u).sum())
                    s[f"infl_g{g}_T{T}_max"] = max(s[f"infl_g{g}_T{T}_max"], float(u.max()))
                    s[f"infl_g{g}_T{T}_n"] += float(full.sum())
        # score margins at the cut: (s_top - s_next) / |s_top|
        sc = scratch.scores[:n][idx]
        live = torch.arange(sc.shape[1], device=dev)[None, :] < complete[:, None]
        vals = torch.where(live, sc, torch.full_like(sc, float("-inf"))).topk(top + 1, dim=1).values
        gap = (vals[:, top - 1] - vals[:, top]) / vals[:, top - 1].abs().clamp_min(1e-30)
        for lim in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
            s[f"gap_lt_{lim:g}"] += float((gap < lim).sum())
        s["gap_zero"] += float((gap == 0).sum())


def summary(sums: dict) -> dict:
    """Per prompt and layer group: means over the summed blocks."""

    out: dict = {}
    for (prompt, layer, bucket), s in sorted(sums.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        rows = s["rows"]
        top_rows = rows * 512
        row = {"rows": int(rows), "recent64": s["recent64"] / top_rows, "recent512": s["recent512"] / top_rows,
               "first64": s["first64"] / top_rows,
               "overlap": s["overlap_sum"] / max(1.0, s["overlap_n"]),
               "gap_zero": s["gap_zero"] / rows}
        for lim in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
            row[f"gap_lt_{lim:g}"] = s[f"gap_lt_{lim:g}"] / rows
        for g in GRAINS:
            for T in TILES:
                n = s[f"infl_g{g}_T{T}_n"]
                if n:
                    mean = s[f"infl_g{g}_T{T}_sum"] / n
                    row[f"infl_g{g}_T{T}"] = mean
                    row[f"infl_g{g}_T{T}_max"] = s[f"infl_g{g}_T{T}_max"]
        out[f"{prompt} | layer {layer} | {bucket // 1024}k"] = row
    return out


def rollup(rows: dict) -> dict:
    """Per prompt and bucket, averaged over the attention layers (rows-weighted)."""

    acc: dict = defaultdict(lambda: defaultdict(float))
    for name, r in rows.items():
        prompt, layer, bucket = [p.strip() for p in name.split("|")]
        if layer == "layer mtp":
            continue
        a = acc[f"{prompt} | {bucket}"]
        w = r["rows"]
        a["rows"] += w
        for k, v in r.items():
            if k != "rows" and not k.endswith("_max"):
                a[k] += v * w
            elif k.endswith("_max"):
                a[k] = max(a[k], v)
    return {k: {kk: (vv / a["rows"] if kk != "rows" and not kk.endswith("_max") else vv) for kk, vv in a.items()}
            for k, a in acc.items()}


def select(args) -> None:
    from tensorfold.families.qwen4_exp.cuda import attention as attn_mod
    from tensorfold.families.qwen4_exp.cuda import forward

    args.out.mkdir(parents=True, exist_ok=True)
    engine = pp.build_engine(args.model, 4, "int8")
    sel = Selections(attn_mod, forward)
    makers = {"syn8k": (8192, False, 4242 + 8192), "syn32k": (32768, False, 4242 + 32768),
              "nat32k": (32768, True, 515151), "syn128k": (131072, False, 4242 + 131072),
              "nat128k": (131072, True, 525252)}
    report: dict = {"prompts": {}}
    for name in [p for p in args.prompts.split(",") if p]:
        n, natural, seed = makers[name]
        ids = pp.prompt_ids(args.model, n, seed, natural=natural)
        nr.forget(engine)
        sel.prompt = name
        t = time.perf_counter()
        got = pp.run_prompt(engine, ids)
        sel.prompt = ""
        report["prompts"][name] = {"tokens": len(ids), "seconds": time.perf_counter() - t,
                                   "logits": got["logits_sha"], "first": got["first"]}
        print(f"[qsa] {name}: {len(ids)} tokens in {time.perf_counter() - t:.1f}s with summaries "
              f"({sel.calls} blocks so far, {sel.seconds:.1f}s summarising)", flush=True)
    rows = summary(sel.sums)
    report["blocks"] = rows
    report["rollup"] = rollup(rows)
    (args.out / "qsa_select.json").write_text(json.dumps(report, indent=1))
    keys = ["recent64", "recent512", "first64", "overlap", "gap_lt_1e-06", "gap_lt_1e-05"]
    keys += [f"infl_g{g}_T{T}" for g in GRAINS for T in TILES]
    print("[qsa] per prompt and 8k bucket, all attention layers (rows-weighted):")
    for name, r in report["rollup"].items():
        print(f"  {name}: rows {int(r['rows'])} " + " ".join(f"{k} {r.get(k, float('nan')):.3f}" for k in keys),
              flush=True)


def timed(fn, reps: int = 5) -> float:
    """The median of ``reps`` CUDA-event timings of ``fn`` (after one warm call), in ms."""

    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]


def dequant64(c: torch.Tensor, s: torch.Tensor | None, keys: torch.Tensor, bits: int) -> torch.Tensor:
    """Cache rows ``keys`` as both kernels see them (bf16 values: the dequant's rounding included), in fp64."""

    if not bits:
        return c[keys].double()
    code = c[keys].float()                                            # [n, HK, D]
    sc = s[keys].float()                                              # [n, HK, D // 32]
    n, hk, d = code.shape
    v = (code.view(n, hk, d // 32, 32) + 0.5) * sc[..., None] * 0.0078125
    return v.view(n, hk, d).to(torch.bfloat16).double()


def h32_64(x: torch.Tensor) -> torch.Tensor:
    """H32 over groups of 32 of the last axis, in fp64 (the cache's rotation; H32 H32 = I)."""

    import math

    hm = torch.ones((1, 1), dtype=torch.float64, device=x.device)
    for _ in range(5):
        hm = torch.cat([torch.cat([hm, hm], 1), torch.cat([hm, -hm], 1)], 0)
    hm = hm / math.sqrt(32)
    shp = x.shape
    return (x.reshape(-1, 32) @ hm.T).reshape(shp)


class Bench:
    """On chosen prompt slices (prompt, attention layer, first row), after TensorFold's own calls: Thor's calls on the
    same inputs, timed with CUDA events and compared: the selections (``qsa_rows``: Thor's tensor-core scores against
    TensorFold's, and both against an fp64 selection on sampled rows) and the attention on TensorFold's selection
    (Thor's shadow path against TensorFold's kernel, both against fp64 on sampled rows). The served path's results
    are restored before it continues."""

    def __init__(self, attn_mod, forward, thor_qsa, plan: dict, sample_rows: int) -> None:
        self.prompt = ""
        self.layer = None
        self.plan = plan
        self.sample_rows = sample_rows
        self.results: list = []
        self.am, self.tq = attn_mod, thor_qsa
        self.cur: dict = {}
        self.shadow = None
        orig_attn, orig_rows, orig_block = attn_mod.attention, attn_mod.qsa_rows, forward.attn_block
        self.orig_attn, self.orig_rows = orig_attn, orig_rows

        def attn_block(layer, w, segs, b, R, mtp, context=None):
            self.layer = "mtp" if mtp else int(layer.index)
            return orig_block(layer, w, segs, b, R, mtp, context=context)

        def chosen(rows, context) -> bool:
            if not self.prompt or context is None:
                return False
            p0 = int(context) - int(rows)
            return p0 in self.plan.get(self.prompt, ()) and self.layer in self.plan.get("layers", ())

        def qsa_rows(iq, pooled, pos0, scratch, rows, *, context=None):
            orig_rows(iq, pooled, pos0, scratch, rows, context=context)
            if chosen(rows, context):
                self.cur = {"prompt": self.prompt, "layer": str(self.layer), "p0": int(context) - int(rows),
                            "rows": int(rows)}
                try:
                    self.rows_bench(iq, pooled, pos0, scratch, int(rows), int(context))
                except Exception as exc:  # noqa: BLE001
                    self.cur["rows_error"] = repr(exc)[-600:]
                    print(f"[qsa] rows bench failed: {repr(exc)[-600:]}", flush=True)
                orig_rows(iq, pooled, pos0, scratch, rows, context=context)      # the served selection again

        def attention(q, kc, vc, pos0, scratch, rows, scale, out=None, *, context=None, ks=None, vs=None, bits=0):
            o = orig_attn(q, kc, vc, pos0, scratch, rows, scale, out=out, context=context, ks=ks, vs=vs, bits=bits)
            if out is not None and chosen(rows, context):
                try:
                    self.attn_bench(q, kc, vc, scratch, int(rows), int(context) - int(rows), scale, out, ks, vs,
                                    bits)
                except Exception as exc:  # noqa: BLE001
                    self.cur["attn_error"] = repr(exc)[-600:]
                    print(f"[qsa] attention bench failed: {repr(exc)[-600:]}", flush=True)
                self.results.append(self.cur)
                print("[qsa] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                          for k, v in self.cur.items()), flush=True)
                self.cur = {}
            return o

        forward.attn_block, attn_mod.attention, attn_mod.qsa_rows = attn_block, attention, qsa_rows

    @torch.no_grad()
    def rows_bench(self, iq, pooled, pos0, scratch, n, context) -> None:
        import triton

        tq, res = self.tq, self.cur
        ratio, top = scratch.ratio, scratch.budget // scratch.ratio
        blocks = min(scratch.nb, max(1, triton.cdiv(context, ratio)))
        ids, sp = scratch.ids[:n].clone(), scratch.sparse[:n].clone()
        sc_ref = scratch.scores[:n, :blocks].clone()
        p0 = context - n
        dev = iq.device
        complete = (p0 + torch.arange(n, device=dev) + 1) // ratio
        live = (torch.arange(blocks, device=dev)[None, :] < complete[:, None]) & (sp[:, None] != 0)
        res["blocks"] = blocks
        res["sparse_rows"] = int((sp != 0).sum())
        res["rows_tf_ms"] = timed(lambda: self.orig_rows(iq, pooled, pos0, scratch, n, context=context))
        if int((sp != 0).sum()):
            res["select_tf_ms"] = timed(lambda: self.am._launch_select(scratch, pos0, n, blocks))
        attn, sel = tq.S.attn, tq.S.sel
        tq.set_attn(False)                                 # THOR-K4-QSA (r19): TensorFold's selection
        tq.set_sel(False)
        res["rows_thor_ms"] = timed(lambda: tq.qsa_rows(iq, pooled, pos0, scratch, n, context=context))
        tq.set_attn(True)
        if not res["sparse_rows"]:
            tq.set_attn(attn)
            tq.set_sel(sel)
            return
        tc = scratch.scores[:n, :blocks].clone()
        ids_tc = scratch.ids[:n].clone()
        nk_tc, sp_tc = scratch.nk[:n].clone(), scratch.sparse[:n].clone()
        listed = (sp_tc != 0)[:, None] & (torch.arange(ids_tc.shape[1], device=dev)[None, :] < nk_tc[:, None])

        def same_lists(fn) -> bool:
            """fn's lists against TensorFold's on Thor's scores, from buffers cleared first (nothing left over)."""
            scratch.ids[:n].fill_(-1)
            scratch.nk[:n].fill_(-1)
            scratch.sparse[:n].fill_(-1)
            fn()
            return bool(torch.equal(scratch.nk[:n], nk_tc) and torch.equal(scratch.sparse[:n], sp_tc)
                        and torch.equal(scratch.ids[:n][listed], ids_tc[listed]))

        res["rows_new_ms"] = timed(lambda: tq.qsa_rows(iq, pooled, pos0, scratch, n, context=context))
        res["select_new_ms"] = timed(lambda: tq.select(scratch, pos0, n, blocks))
        res["select_new_equal"] = same_lists(lambda: tq.select(scratch, pos0, n, blocks))
        width = max(16, triton.next_power_of_2(blocks))
        for w in [w for w in (2, 4, 8, 16) if width <= 4096 * w]:         # at most 128 keys a thread
            res[f"select_new_ms_w{w}"] = timed(lambda: tq.select(scratch, pos0, n, blocks, warps=w))
            res["select_new_equal"] &= same_lists(lambda: tq.select(scratch, pos0, n, blocks, warps=w))
        tq.set_sel(True)                                   # THOR-K4-QSA-SEL: the CUDA radix select
        try:
            res["rows_sel_ms"] = timed(lambda: tq.qsa_rows(iq, pooled, pos0, scratch, n, context=context))
            res["select_sel_equal"] = True
            for v in (tq.SEL_VARIANT,):
                name = "select_sel_ms"
                res[name] = timed(lambda: tq.select(scratch, pos0, n, blocks, variant=v))
                res["select_sel_equal"] &= same_lists(lambda: tq.select(scratch, pos0, n, blocks, variant=v))
        finally:
            tq.set_attn(attn)
            tq.set_sel(sel)
        res["scores_equal_frac"] = float((tc == sc_ref)[live].float().mean())
        res["scores_rel_max"] = float(((tc - sc_ref).abs() / sc_ref.abs().clamp_min(1e-6))[live].max())
        rows_sp = torch.nonzero(sp != 0).squeeze(1)
        a = (ids[rows_sp, 0:top * ratio:ratio] // ratio).long()
        b2 = (ids_tc[rows_sp, 0:top * ratio:ratio] // ratio).long()
        same = (a == b2).all(1)
        res["select_rows"] = int(rows_sp.numel())
        res["select_rows_same"] = int(same.sum())
        pick = rows_sp[torch.linspace(0, rows_sp.numel() - 1, min(32, rows_sp.numel()),
                                      device=dev).round().long().unique()]
        agree_tf = agree_th = 0
        for r in pick.tolist():
            c = int(complete[r])
            s64 = torch.relu(iq[r].double() @ pooled[:c].double().T).sum(0) / (pooled.shape[1] ** 0.5)
            want = torch.sort(torch.sort(-s64, stable=True).indices[:top]).values
            agree_tf += int(torch.equal(want, (ids[r, 0:top * ratio:ratio] // ratio).long()))
            agree_th += int(torch.equal(want, (ids_tc[r, 0:top * ratio:ratio] // ratio).long()))
        res["select64_rows"] = int(pick.numel())
        res["select64_agree_tf"] = agree_tf
        res["select64_agree_thor"] = agree_th

    @torch.no_grad()
    def attn_bench(self, q, kc, vc, scratch, n, p0, scale, served, ks, vs, bits) -> None:
        from types import SimpleNamespace

        tq, res = self.tq, self.cur
        dev = q.device
        ctx = p0 + n
        ref = served[:n].clone()
        pos_t = torch.tensor([p0], dtype=torch.int32, device=dev)
        tmp = torch.empty_like(ref)
        cache = SimpleNamespace(k=kc, v=vc, ks=ks, vs=vs)
        res["attn_tf_ms"] = timed(lambda: self.orig_attn(q, kc, vc, pos_t, scratch, n, scale, out=tmp, context=ctx,
                                                           ks=ks, vs=vs, bits=bits))
        res["attn_tf_rerun_equal"] = bool(torch.equal(tmp, ref))
        shadow = None
        if bits == 8:
            if self.shadow is None or self.shadow.k.shape[0] < ctx:
                self.shadow = tq.Shadow(-(-ctx // 8192) * 8192, kc.shape[1], kc.shape[2], dev, scratch.ratio)
            shadow = self.shadow

            def staged():
                tq.begin(shadow)
                tq.stage(cache, shadow, scratch, pos_t, n, ctx)

            res["stage_ms"] = timed(staged)                    # the slice's blocks, as the first of a segment
        tq.set_attn(False)                                 # THOR-K4-QSA (r19): chunks and merge on the shadow
        res["attn_thor_ms"] = timed(lambda: tq.attention(q[:n], cache, shadow, pos_t, scratch, n, scale, tmp,
                                                         context=ctx, bits=bits))
        thor = tmp.clone()
        tq.set_attn(True)
        res["attn_new_ms"] = timed(lambda: tq.attention(q[:n], cache, shadow, pos_t, scratch, n, scale, tmp,
                                                        context=ctx, bits=bits))
        new = tmp.clone()
        kb, vb = (shadow.k, shadow.v) if bits else (kc, vc)
        for bt, w, st in ((64, 4, 2), (64, 4, 4), (64, 8, 3), (32, 4, 3), (32, 4, 5), (32, 2, 4), (128, 8, 2),
                          (128, 8, 3)):
            try:
                res[f"attn_new_ms_{bt}_{w}_{st}"] = timed(lambda: tq.row_attention(
                    q[:n], kb, vb, pos_t, scratch, n, scale, tmp, bits=bits, tile=bt, warps=w, stages=st))
                res[f"attn_new_equal_{bt}_{w}_{st}"] = bool(torch.equal(tmp, new)) if bt == tq.ROW_TILE else None
            except Exception as exc:  # noqa: BLE001
                res[f"attn_new_ms_{bt}_{w}_{st}"] = repr(exc)[-200:]
        res["attn_equal_frac"] = float((thor == ref).float().mean())
        res["attn_rel"] = float((thor.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-30))
        pick = torch.linspace(0, n - 1, min(self.sample_rows, n), device=dev).round().long().unique()
        errs_tf, errs_th, errs_new = [], [], []
        h, d = q.shape[1], q.shape[2]
        hk = kc.shape[1]
        g = h // hk
        for r in pick.tolist():
            end = p0 + r + 1
            if scratch.qsa and int(scratch.sparse[r]) != 0:
                keys = scratch.ids[r, :int(scratch.nk[r])].long()
            else:
                keys = torch.arange(end, device=dev)
            kk = dequant64(kc, ks, keys, bits)
            vv = dequant64(vc, vs, keys, bits)
            qq = q[r].double()
            want = torch.empty((h, d), dtype=torch.float64, device=dev)
            for j in range(hk):
                sc = qq[j * g:(j + 1) * g] @ kk[:, j].T * scale
                want[j * g:(j + 1) * g] = torch.softmax(sc, dim=1) @ vv[:, j]
            if bits:
                want = h32_64(want)
            nrm = want.norm(dim=1).clamp_min(1e-30)
            errs_tf.append(((ref[r].double() - want).norm(dim=1) / nrm).max().item())
            errs_th.append(((thor[r].double() - want).norm(dim=1) / nrm).max().item())
            errs_new.append(((new[r].double() - want).norm(dim=1) / nrm).max().item())
        res["err_rows"] = len(errs_tf)
        res["err_tf_mean"] = sum(errs_tf) / len(errs_tf)
        res["err_thor_mean"] = sum(errs_th) / len(errs_th)
        res["err_tf_max"] = max(errs_tf)
        res["err_thor_max"] = max(errs_th)
        res["err_ratio_max"] = max(b / a for a, b in zip(errs_tf, errs_th))
        res["err_new_mean"] = sum(errs_new) / len(errs_new)
        res["err_new_max"] = max(errs_new)
        res["err_ratio_new_max"] = max(b / a for a, b in zip(errs_tf, errs_new))
        res["err_ratio_new_mean"] = res["err_new_mean"] / res["err_tf_mean"]


PROMPTS = {"syn8k": (8192, False, 4242 + 8192), "syn32k": (32768, False, 4242 + 32768),
           "nat32k": (32768, True, 515151), "syn128k": (131072, False, 4242 + 131072),
           "nat128k": (131072, True, 525252)}


def kernels(args) -> None:
    from tensorfold.families.qwen4_exp.cuda import attention as attn_mod
    from tensorfold.families.qwen4_exp.cuda import forward
    from tensorfold.families.qwen4_exp.cuda import thor_qsa

    args.out.mkdir(parents=True, exist_ok=True)
    engine = pp.build_engine(args.model, 4, "int8")
    thor_qsa.set_active(False)                          # the served path is TensorFold's; Thor's runs beside it
    layers = sorted({int(layer.index) for layer in engine.w.layers if not layer.linear})
    pick_layers = [layers[0], layers[len(layers) // 2], layers[-1]]
    plan: dict = {"layers": pick_layers}
    for name, (n, _, _) in PROMPTS.items():
        plan[name] = [p - 256 for p in (2048, 2304, 4096, 8192, 16384, 32768, 65536, 131072) if p <= n]
    bench = Bench(attn_mod, forward, thor_qsa, plan, args.sample_rows)
    print(f"[qsa] attention layers {layers}; sampled {pick_layers}", flush=True)
    for name in [p for p in args.prompts.split(",") if p]:
        n, natural, seed = PROMPTS[name]
        ids = pp.prompt_ids(args.model, n, seed, natural=natural)
        nr.forget(engine)
        bench.prompt = name
        got = pp.run_prompt(engine, ids)
        bench.prompt = ""
        print(f"[qsa] {name}: done ({got['tok_s']:.0f} tok/s with the benches inside)", flush=True)
    (args.out / "qsa_kernels.json").write_text(json.dumps(bench.results, indent=1))


def selall(args) -> None:
    """THOR-K4-QSA-SEL: in served prompt passes (the Thor path on), every slice's selection (every attention layer and
    256-row block) against TensorFold's ``_launch_select`` on the same scores; lists, key counts and sparse flags."""

    from types import SimpleNamespace

    from tensorfold.families.qwen4_exp.cuda import attention as attn_mod
    from tensorfold.families.qwen4_exp.cuda import thor_qsa

    args.out.mkdir(parents=True, exist_ok=True)
    engine = pp.build_engine(args.model, 4, "int8")
    orig = thor_qsa.select
    st = {"sel": bool(thor_qsa.S.sel), "attn": bool(thor_qsa.S.attn), "use": bool(thor_qsa.use()), "calls": 0,
          "rows": 0, "sparse_rows": 0, "mismatch_calls": 0, "mismatch_rows": 0, "max_blocks": 0, "first": []}
    ref = {}

    def select(scratch, pos0, rows, blocks, warps=None, variant=None):
        orig(scratch, pos0, rows, blocks, warps=warps, variant=variant)
        key = id(scratch)
        if key not in ref:
            ref[key] = SimpleNamespace(scores=scratch.scores, ids=torch.empty_like(scratch.ids),
                                       nk=torch.empty_like(scratch.nk), sparse=torch.empty_like(scratch.sparse),
                                       nb=scratch.nb, ratio=scratch.ratio, budget=scratch.budget, idw=scratch.idw)
        r = ref[key]
        r.ids.fill_(-1)
        attn_mod._launch_select(r, pos0, rows, blocks)
        nk, sp = scratch.nk[:rows], scratch.sparse[:rows]
        listed = (r.sparse[:rows] != 0)[:, None] & (torch.arange(r.ids.shape[1], device=nk.device)[None, :]
                                                     < r.nk[:rows, None])
        bad = (nk != r.nk[:rows]) | (sp != r.sparse[:rows]) | ((scratch.ids[:rows] != r.ids[:rows]) & listed).any(1)
        nbad = int(bad.sum())
        st["calls"] += 1
        st["rows"] += int(rows)
        st["sparse_rows"] += int((r.sparse[:rows] != 0).sum())
        st["max_blocks"] = max(st["max_blocks"], int(blocks))
        if nbad:
            st["mismatch_calls"] += 1
            st["mismatch_rows"] += nbad
            if len(st["first"]) < 8:
                st["first"].append({"call": st["calls"], "pos0": int(pos0[0]), "rows": int(rows), "blocks": int(blocks),
                                    "row": int(torch.nonzero(bad)[0, 0])})

    thor_qsa.select = select
    per = {}
    for name in [p for p in args.prompts.split(",") if p]:
        n, natural, seed = PROMPTS[name]
        ids = pp.prompt_ids(args.model, n, seed, natural=natural)
        nr.forget(engine)
        before = dict(st)
        got = pp.run_prompt(engine, ids)
        per[name] = {k: st[k] - before[k] for k in ("calls", "rows", "sparse_rows", "mismatch_calls", "mismatch_rows")}
        print(f"[selall] {name}: {per[name]} ({got['tok_s']:.0f} tok/s with the checks inside)", flush=True)
    thor_qsa.select = orig
    st["per_prompt"] = per
    (args.out / "qsa_selall.json").write_text(json.dumps(st, indent=1))
    print(f"[selall] sel {st['sel']} attn {st['attn']} use {st['use']}: {st['calls']} calls, {st['sparse_rows']} "
          f"sparse rows, max blocks {st['max_blocks']}, mismatching rows {st['mismatch_rows']}", flush=True)


def two_streams(engine, a: list[int], b: list[int]) -> tuple:
    """Two prompts admitted together (their passes packed), one token each: each one's head-logit hash."""

    import hashlib

    from tensorfold.cuda.streams import Stream
    from tensorfold.families.qwen4_exp.cuda import multi as multi_mod

    multi = engine.multi
    md = multi_mod.MultiDecoder
    joined = md._joined
    seen: dict = {}

    def hashed(dec, pieces, heads, lasts, spent):
        head = 0                                       # ``_joined``'s order: a head row for each prompt that ends
        for st, a0, n0 in pieces:
            if a0 + n0 >= len(st.prompt) and heads is not None:
                seen[st.sid] = heads[head:head + 1].detach().clone()
                head += 1
        return joined(dec, pieces, heads, lasts, spent)

    md._joined = hashed
    try:
        sa, sb = Stream(list(a), 1, None, draft=True), Stream(list(b), 1, None, draft=True)
        multi.admit(sa)
        multi.admit(sb)
        while not (sa.done and sb.done):
            multi.finish(multi.round())
    finally:
        md._joined = joined
    return tuple(hashlib.sha256(seen[s.sid].view(torch.int16).cpu().numpy().tobytes()).hexdigest()[:16]
                 if s.sid in seen else None for s in (sa, sb))


def ab(args) -> None:
    """One engine as served, TensorFold's prompt QSA (``tf``) and Thor's (``thor``) switched per run."""

    from tensorfold.families.qwen4_exp.cuda import thor_qsa
    from tensorfold.families.qwen4_exp.cuda.decode import entry_end

    args.out.mkdir(parents=True, exist_ok=True)
    engine = pp.build_engine(args.model, 4, "int8")
    if not thor_qsa.use():
        raise SystemExit("thor_qsa is not on in this process (not sm_110, or TENSORFOLD_THOR_QSA=0)")
    w = engine.w
    pp.instrument(w, w.cfg)
    table = nr.Table(w)
    report: dict = {"runs": [], "events": {}, "checks": {}}
    runs = report["runs"]
    syn = {n: pp.prompt_ids(args.model, n, 4242 + n) for n in (8192, 32768, 131072)}
    nat = pp.prompt_ids(args.model, 32768, 515151, natural=True)
    prompts = (("synthetic 8k", syn[8192], False), ("synthetic 32k", syn[32768], False),
               ("natural 32k warm", nat, False), ("natural 32k cold", nat, True),
               ("synthetic 128k", syn[131072], False))

    def events(mode: str, n: int) -> None:
        thor_qsa.set_active(mode == "thor")
        nr.forget(engine)
        pp.L.reset("events")
        got = pp.run_prompt(engine, syn[n])
        summary = pp.events_summary(pp.L.marks)
        paths = summary["paths"]
        attn = {}
        for k, v in paths.items():
            leaf = k.split("/")[-1]
            if leaf.startswith("attn"):
                attn[leaf] = attn.get(leaf, 0.0) + v
        cls = summary["classes"]
        report["events"][f"{mode} {n}"] = {"tok_s": got["tok_s"], "classes": cls, "attn": attn,
                                           "total_ms": summary["total_ms"], "logits": got["logits_sha"]}
        pp.L.marks = []
        pp.L.reset("plain")
        print(f"[qsa] events {n} {mode}: {got['tok_s']:.0f} tok/s; QSA class {cls.get('QSA / attention', 0):.0f} ms "
              f"of {summary['total_ms']:.0f}; " + ", ".join(f"{k} {v:.0f}" for k, v in sorted(attn.items())),
              flush=True)

    def plain(mode: str, order: int) -> None:
        thor_qsa.set_active(mode == "thor")
        for label, ids, cold in prompts:
            extra = {"mode": mode, "order": order}
            if cold:
                extra["drop"] = table.drop()
            nr.run(engine, ids, f"{label} {mode} {order}", table, runs, extra)

    def followup(mode: str, n: int, seed: int) -> None:
        """An n-token prompt, then the same prompt and 40 new tokens, resumed from the first's kept point."""

        thor_qsa.set_active(mode == "thor")
        nr.forget(engine)
        base = pp.prompt_ids(args.model, n, seed)
        pp.run_prompt(engine, base)
        got = pp.run_prompt(engine, base + pp.prompt_ids(args.model, 512, seed + 1)[-40:])
        report.setdefault("followup", {}).setdefault(f"{mode} {n}", []).append(
            {"seconds": got["seconds"], "kept_tokens": entry_end(base), "logits": got["logits_sha"]})
        print(f"[qsa] follow-up {mode} {n}: {got['tokens']} tokens resumed in {got['seconds']:.3f}s (kept prefix "
              f"{entry_end(base)})", flush=True)

    for n in [int(v) for v in args.events.split(",") if v]:
        for mode in ("tf", "thor"):
            events(mode, n)
    for n in (16384, 32768, 61440):
        for i, mode in enumerate(("tf", "thor", "thor", "tf")):
            followup(mode, n, 777 + 10 * i + n)
    # invariants on this path (thor): a resumed prompt equals a fresh one; packed equals solo
    thor_qsa.set_active(True)
    base = pp.prompt_ids(args.model, 24576, 9901)
    longer = base + pp.prompt_ids(args.model, 512, 9902)[-300:]
    nr.forget(engine)
    pp.run_prompt(engine, base)
    kept = entry_end(base)
    resumed = pp.run_prompt(engine, longer)
    nr.forget(engine)
    fresh = pp.run_prompt(engine, longer)
    report["checks"]["resumed == fresh (thor, kept prefix %s)" % kept] = resumed["logits_sha"] == fresh["logits_sha"]
    a_ids, b_ids = pp.prompt_ids(args.model, 12288, 9911), pp.prompt_ids(args.model, 20480, 9912, natural=True)
    nr.forget(engine)
    solo_a = pp.run_prompt(engine, a_ids)["logits_sha"]
    nr.forget(engine)
    solo_b = pp.run_prompt(engine, b_ids)["logits_sha"]
    nr.forget(engine)
    both = two_streams(engine, a_ids, b_ids)
    report["checks"]["packed == solo (thor), first prompt"] = both[0] == solo_a
    report["checks"]["packed == solo (thor), second prompt"] = both[1] == solo_b
    plain("tf", 1)
    plain("thor", 1)
    plain("thor", 2)
    plain("tf", 2)
    thor_qsa.set_active(True)
    by = {r["label"]: r for r in runs}
    table_rows = {}
    for label, _, _ in prompts:
        s1, t1, t2, s2 = (by[f"{label} {m}"] for m in ("tf 1", "thor 1", "thor 2", "tf 2"))
        table_rows[label] = {"tf": [s1["tok_s"], s2["tok_s"]], "thor": [t1["tok_s"], t2["tok_s"]],
                             "gain_first": t1["tok_s"] / s1["tok_s"] - 1, "gain_second": t2["tok_s"] / s2["tok_s"] - 1,
                             "first_token_as_tf": t1["first"] == s1["first"]}
        report["checks"][f"{label}: thor deterministic"] = t1["logits"] == t2["logits"]
        report["checks"][f"{label}: tf deterministic"] = s1["logits"] == s2["logits"]
        print(f"[qsa] {label}: tf {s1['tok_s']:.0f}/{s2['tok_s']:.0f}, thor {t1['tok_s']:.0f}/{t2['tok_s']:.0f} tok/s "
              f"({100 * table_rows[label]['gain_first']:+.1f}% / {100 * table_rows[label]['gain_second']:+.1f}%); "
              f"first token as tf {table_rows[label]['first_token_as_tf']}", flush=True)
    report["table"] = table_rows
    for k, v in report["checks"].items():
        print(f"[qsa] check {k}: {v}", flush=True)
    (args.out / "qsa_ab.json").write_text(json.dumps(report, indent=1))
    metrics = {"prefill_8k": by["synthetic 8k thor 1"]["tok_s"], "prefill_32k": by["synthetic 32k thor 1"]["tok_s"],
               "prefill_128k": by["synthetic 128k thor 1"]["tok_s"],
               "tests_passed": sum(1 for v in report["checks"].values() if v),
               "tests_failed": sum(1 for v in report["checks"].values() if not v)}
    (args.out / "ledger_metrics.json").write_text(json.dumps(metrics))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("model", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--prompts", default="syn32k,nat32k,syn128k")
    k = sub.add_parser("kernels")
    k.add_argument("model", type=Path)
    k.add_argument("out", type=Path)
    k.add_argument("--prompts", default="nat32k,syn128k")
    k.add_argument("--sample-rows", type=int, default=24)
    c = sub.add_parser("selall")
    c.add_argument("model", type=Path)
    c.add_argument("out", type=Path)
    c.add_argument("--prompts", default="nat32k,syn128k")
    a = sub.add_parser("ab")
    a.add_argument("model", type=Path)
    a.add_argument("out", type=Path)
    a.add_argument("--events", default="8192,32768,131072")
    args = ap.parse_args()
    if args.cmd == "select":
        select(args)
    elif args.cmd == "kernels":
        kernels(args)
    elif args.cmd == "selall":
        selall(args)
    elif args.cmd == "ab":
        ab(args)


if __name__ == "__main__":
    main()
