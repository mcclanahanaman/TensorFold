"""A QSA-aware referee for G-B (THOR-K4-QSA): every prompt pass's sparse attention computed exactly.

Both steps of Flash Next's QSA, from the values both runs read:

- the indexer: each sparse row's block scores in float64 (relu(q . k) for each indexer head, summed over the heads,
  over sqrt(index dim); the queries and the pooled keys as the model stores them, bf16), then TensorFold's selection
  rule on them: the top ``budget / ratio`` blocks, ties to lower block ids, in block order, then the row's
  unfinished tail;
- the attention: each row over its keys (its selected blocks and tail, or, for a dense row, every key up to its own)
  in float64: the cache's values as the model stores them (bf16, or the 8-bit cache's dequantized bf16 values, still
  rotated, against the query in the same rotation), the softmax and the weighted sum in float64, the rotation undone
  in float64, rounded once to bf16 (``dense_reference.bf16_rounded``), the dtype of the buffer it fills.

Neither TensorFold's kernels (fp32 CUDA-core block scores, bf16 softmax weights, 512-key partials) nor Thor's
(tensor-core block scores, the bf16 shadow) decide a near-tie, so the referee sides with neither by construction.
Thor's prompt QSA path is switched off in the process (prompt passes call ``attention.qsa_rows`` and
``attention.attention``, both replaced here); decode windows keep TensorFold's attention, on these selections.
The rest of a prompt pass is TensorFold's own, or the other referees' (moe_reference, dense_reference,
gdn_reference). The first block of a process is also run through TensorFold's kernels and compared, as a wiring
check: a selection agreeing on under 90% of the sparse rows, or an attention output further than 5% (of the largest
value) from TensorFold's, stops the run.

    import qsa_reference; qsa_reference.install()
"""

from __future__ import annotations

import math

import torch

ROWS = 32                    # rows a float64 attention product takes at once
_INSTALLED: dict = {}
CHECK: dict = {}             # the wiring check's results, once


def _h32(x: torch.Tensor) -> torch.Tensor:
    """H32 over groups of 32 of the last axis, in float64 (the cache's rotation; H32 H32 = I)."""

    hm = torch.ones((1, 1), dtype=torch.float64, device=x.device)
    for _ in range(5):
        hm = torch.cat([torch.cat([hm, hm], 1), torch.cat([hm, -hm], 1)], 0)
    hm = hm / math.sqrt(32)
    return (x.reshape(-1, 32) @ hm.T).reshape(x.shape)


def _values(cache: torch.Tensor, scales: torch.Tensor | None, n: int, bits: int) -> torch.Tensor:
    """Cache rows [0, n) as float64 [n, HK, D]: bf16 as stored, or 8-bit codes dequantized to the bf16 values the
    kernels read (``kvquant.dequant_group_8``: (code + 0.5) * s / 128, rounded to bf16), still rotated."""

    if not bits:
        return cache[:n].double()
    if bits != 8:
        raise ValueError("the QSA referee reads bf16 or 8-bit caches")
    code = cache[:n].float()
    s = scales[:n].float()
    rows, hk, d = code.shape
    v = (code.view(rows, hk, d // 32, 32) + 0.5) * s[..., None] * 0.0078125
    return v.view(rows, hk, d).to(torch.bfloat16).double()


def select(iq: torch.Tensor, pooled: torch.Tensor, p0: int, scratch, rows: int) -> None:
    """Rows p0 .. p0 + rows - 1: each sparse row's blocks from float64 scores into ``scratch`` (ids, nk, sparse), in
    TensorFold's format."""

    ratio, top = scratch.ratio, scratch.budget // scratch.ratio
    dev = iq.device
    end = p0 + torch.arange(rows, device=dev) + 1
    complete = end // ratio
    sparse = complete > top
    scratch.nk[:rows] = torch.where(sparse, top * ratio + end - ratio * complete, end).to(torch.int32)
    scratch.sparse[:rows] = sparse.to(torch.int32)
    idx = torch.nonzero(sparse).squeeze(1)
    if idx.numel() == 0:
        return
    cmax = int(complete[idx].max())
    keys = pooled[:cmax].double()                                                 # [blocks, DI]
    q = iq[idx].double()                                                          # [sparse rows, HI, DI]
    s = torch.relu(torch.einsum("rhd,bd->rhb", q, keys)).sum(1) / math.sqrt(pooled.shape[1])
    live = torch.arange(cmax, device=dev)[None, :] < complete[idx][:, None]
    s = torch.where(live, s, torch.full_like(s, -math.inf))
    chosen = torch.sort(torch.sort(-s, dim=1, stable=True).indices[:, :top], dim=1).values      # ties: lower ids
    ids = (chosen[:, :, None] * ratio + torch.arange(ratio, device=dev)).reshape(idx.numel(), top * ratio)
    scratch.ids[idx, :top * ratio] = ids.to(torch.int32)
    tail = ratio * complete[idx][:, None] + torch.arange(ratio, device=dev)[None, :]
    held = scratch.ids[idx, top * ratio:top * ratio + ratio]
    scratch.ids[idx, top * ratio:top * ratio + ratio] = torch.where(tail < end[idx][:, None], tail.to(torch.int32),
                                                                    held)


def attend(q: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor, p0: int, scratch, rows: int, scale: float,
           out: torch.Tensor, ks: torch.Tensor | None, vs: torch.Tensor | None, bits: int) -> torch.Tensor:
    """Rows p0 .. p0 + rows - 1 over their keys in float64 (the selections in ``scratch``), rounded once to bf16
    into ``out``."""

    from dense_reference import bf16_rounded

    dev = q.device
    hk = kc.shape[1]
    g = q.shape[1] // hk
    ctx = p0 + rows
    keys = _values(kc, ks, ctx, bits)                                             # [ctx, HK, D] float64
    vals = _values(vc, vs, ctx, bits)
    width = scratch.ids.shape[1]
    for r0 in range(0, rows, ROWS):
        n = min(ROWS, rows - r0)
        end = p0 + r0 + torch.arange(n, device=dev) + 1
        mask = torch.arange(ctx + 1, device=dev)[None, :] < end[:, None]          # dense rows: every key to theirs
        if scratch.qsa:
            sp = scratch.sparse[r0:r0 + n] != 0
            if bool(sp.any()):
                ids = scratch.ids[r0:r0 + n].long()
                # the sparse rows' lists only: a dense row's nk is its length, and its slots still hold an earlier
                # prompt's selections (keys past this prompt's end: the gate run's out-of-bounds scatter)
                ok = (torch.arange(width, device=dev)[None, :] < scratch.nk[r0:r0 + n, None].long()) & sp[:, None]
                bad = ok & ((ids < 0) | (ids >= end[:, None]))
                if bool(bad.any()):
                    row = int(torch.nonzero(bad.any(1))[0, 0])
                    raise RuntimeError(f"qsa_reference: row {p0 + r0 + row}'s key list holds a key past its end "
                                       f"({int(ids[row][bad[row]][0])} of {int(end[row])} keys)")
                sel = torch.zeros((n, ctx + 1), dtype=torch.bool, device=dev)
                sel.scatter_(1, torch.where(ok, ids, torch.full_like(ids, ctx)), True)   # column ctx: unused slots
                mask = torch.where(sp[:, None], sel, mask)
        mask = mask[:, :ctx]
        cols = torch.nonzero(mask.any(0)).squeeze(1)                              # the keys any of these rows read
        m = mask[:, cols]
        for j in range(hk):
            qq = q[r0:r0 + n, j * g:(j + 1) * g].double()                         # [n, g, D]
            s = torch.einsum("ngd,cd->ngc", qq, keys[cols, j]) * scale
            s = s.masked_fill(~m[:, None, :], -math.inf)
            o = torch.einsum("ngc,cd->ngd", torch.softmax(s, dim=-1), vals[cols, j])
            if bits:
                o = _h32(o)
            out[r0:r0 + n, j * g:(j + 1) * g] = bf16_rounded(o)
    return out


def install() -> None:
    """Prompt passes' QSA exact in this process (Thor's prompt QSA path off)."""

    from tensorfold.families.qwen4_exp.cuda import attention as attn_mod
    from tensorfold.families.qwen4_exp.cuda import thor_qsa

    if _INSTALLED:
        return
    thor_qsa.set_active(False)
    orig_rows, orig_attn = attn_mod.qsa_rows, attn_mod.attention

    def qsa_rows(iq, pooled, pos0, scratch, rows, *, context=None):
        p0 = int(pos0.reshape(-1)[0])
        if "select" not in CHECK:                                                 # the wiring check, once
            orig_rows(iq, pooled, pos0, scratch, rows, context=context)
            tf_ids, tf_sp = scratch.ids[:rows].clone(), scratch.sparse[:rows].clone()
        select(iq, pooled, p0, scratch, int(rows))
        if "select" not in CHECK and bool((tf_sp != 0).any()):
            top = scratch.budget // scratch.ratio * scratch.ratio
            idx = torch.nonzero(tf_sp != 0).squeeze(1)
            same = (scratch.ids[idx, :top] == tf_ids[idx, :top]).all(1).float().mean().item()
            CHECK["select"] = same
            print(f"[qsa_reference] wiring check: float64 selections equal TensorFold's in {100 * same:.2f}% of "
                  f"{idx.numel()} sparse rows", flush=True)
            if same < 0.9:
                raise RuntimeError(f"qsa_reference: selections agree with TensorFold's on {same:.3f} of the rows")

    def attention(q, kc, vc, pos0, scratch, rows, scale, out=None, *, context=None, ks=None, vs=None, bits=0):
        if out is None:                                                           # a decode window: TensorFold's
            return orig_attn(q, kc, vc, pos0, scratch, rows, scale, out=out, context=context, ks=ks, vs=vs,
                             bits=bits)
        p0 = int(pos0.reshape(-1)[0])
        attend(q, kc, vc, p0, scratch, int(rows), scale, out, ks, vs, bits)
        if "attention" not in CHECK:
            tf = torch.empty_like(out[:rows])
            orig_attn(q, kc, vc, pos0, scratch, rows, scale, out=tf, context=context, ks=ks, vs=vs, bits=bits)
            got = out[:rows].float()
            rel = float((tf.float() - got).abs().max()) / max(1e-30, float(got.abs().max()))
            CHECK["attention"] = rel
            print(f"[qsa_reference] wiring check: TensorFold's attention within {rel:.2e} (of the largest value) of "
                  f"the float64 attention on {int(rows)} rows from position {p0}", flush=True)
            if rel > 0.05:
                raise RuntimeError(f"qsa_reference: the float64 attention is {rel:.3f} from TensorFold's kernel")
        return out

    attn_mod.qsa_rows, attn_mod.attention = qsa_rows, attention
    _INSTALLED["on"] = True
