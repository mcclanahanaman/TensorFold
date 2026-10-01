"""A dense-aware referee for G-B (THOR-K2): Flash Next's dense prompt GEMMs computed exactly from the exact weights.

Every dense GEMM of a prompt pass -- the DeltaNet and attention in/out projections, the shared expert, both
hyper-connection projections, the n-gram key/value, the MTP head's fc, the router and the head -- multiplies x (bf16,
exact) by the exact weight (MXFP8: e4m3 x 2^(e - 127), decoded by thor_dense's tested kernel; bf16 as stored)
exactly -- int8 slices of both on the tensor cores, summed in int32 and combined in float64 (``ozaki``: to 2^-49 of
each row's largest value; float64 cuBLAS is about 40x slower on Thor; the first call of each shape is checked against
it) -- and rounds once to the dtype its buffer holds: bf16 rounded correctly (round-to-odd to fp32, then to the
nearest bf16), or fp32 (the hyper-connection down projection's sums, the router's logits). With moe_reference's fp32
routed experts every GEMM of the forward is then exact up to its storage; the rest (attention, the DeltaNet
recurrence, norms, the shared expert's SwiGLU, KV quantization) is TensorFold's own arithmetic, which both runs share.
No path's GEMM summation order survives, so the referee sides with neither by construction. The head's float64 logits
(before their bf16 rounding) are kept in ``LAST["head"]``.

    import dense_reference; dense_reference.install(weights)     # prompt passes (the HC down projection: always)
"""

from __future__ import annotations

import torch

LAST: dict = {}
HEAD_ROWS = 32768            # vocabulary rows of the head a float64 product takes at once
_INSTALLED: dict = {}


def bf16_rounded(v: torch.Tensor) -> torch.Tensor:
    """float64 -> bf16, rounded once to the nearest (even): round-to-odd to fp32 first, then fp32 -> bf16."""

    f = v.float()
    toward_zero = torch.where(f.double().abs() > v.abs(), torch.nextafter(f, torch.zeros_like(f)), f)
    odd = toward_zero.view(torch.int32) | (toward_zero.double() != v).to(torch.int32)
    return odd.view(torch.float32).to(torch.bfloat16)


def exact(q, lo: int = 0, hi: int | None = None) -> torch.Tensor:
    """Rows lo:hi of a weight in float64 [N, K]: MXFP8 decoded exactly, bf16 as stored, a router's rows, FP4 tables."""

    if getattr(q, "w8", None) is not None:                           # Mx8Linear
        from tensorfold.cuda.nvfp4 import thor_dense

        return thor_dense.dequantize(q).double()[lo:hi]
    if isinstance(q, torch.Tensor):
        return q[lo:hi].double()
    inner = getattr(q, "b", None)
    weight = inner.weight if inner is not None else getattr(q, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.dtype == torch.bfloat16:
        return weight[lo:hi].double()
    from tensorfold.families.qwen4_exp.cuda import nvfp4

    if isinstance(q, nvfp4.FP4):
        return nvfp4.dequantize_fp4(q).double()[lo:hi]
    raise TypeError(f"no exact weight for {type(q).__name__}")


SLICES = 7                   # int8 slices an operand row is cut into: 49 bits below its largest value
MODE = "ozaki"               # "ozaki" (int8 tensor cores, exact sums) or "fp64" (cuBLAS DGEMM, about 40x slower)


def _slices(v: torch.Tensor, count: int = SLICES) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Rows of v (exact in float64) as 2^e [R, 1] times sum_s 2^(-7 (s + 1)) * S_s, S_s int8: v's every bit down to
    49 below its row's largest value (bf16 and MXFP8 values hold 8 or fewer significant bits)."""

    a = v.double()
    amax = a.abs().amax(dim=1, keepdim=True)
    e = torch.where(amax > 0, torch.floor(torch.log2(amax)) + 1, torch.zeros_like(amax))
    r = a * torch.exp2(-e)                                       # |r| < 1, exact
    out = []
    for _ in range(count):
        r = r * 128.0
        q = torch.trunc(r)                                       # |q| <= 127
        out.append(q.to(torch.int8))
        r = r - q                                                # exact
    return e, out


def _pad8(t: torch.Tensor, least: int = 8) -> torch.Tensor:
    rows = max(least, -(-t.shape[0] // 8) * 8)
    return t if rows == t.shape[0] else torch.cat([t, t.new_zeros((rows - t.shape[0], t.shape[1]))])


def ozaki(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x [M, K] @ w [N, K].T in float64 from int8 slice products summed exactly in int32 (|sum| <= K 127^2 < 2^31)
    and combined in float64: the exact product of bf16-exact operands, to 2^-49 of the operands' row maxima."""

    m, n = x.shape[0], w.shape[0]
    ex, xs = _slices(_pad8(x, 24))
    ew, ws = _slices(_pad8(w))
    y = torch.zeros((xs[0].shape[0], ws[0].shape[0]), dtype=torch.float64, device=x.device)
    for level in range(SLICES - 1, -1, -1):                      # the smallest terms first
        acc = None                                               # a level's sums: at most 7 K 127^2 < 2^31, exact
        for s in range(level + 1):
            c = torch._int_mm(xs[s], ws[level - s].t())
            acc = c if acc is None else acc.add_(c)
        y += acc.double() * 2.0 ** (-7 * (level + 2))
    return (y * torch.exp2(ex) * torch.exp2(ew).t())[:m, :n]


CHECKED: dict = {}           # (M, N, K) -> the first such call's max relative difference from float64 cuBLAS


def _mul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    if MODE != "ozaki":
        return x.double() @ w.double().t()
    y = ozaki(x, w)
    key = (x.shape[0], w.shape[0], x.shape[1])
    if key not in CHECKED:                                        # the first call of a shape: against float64 cuBLAS
        ref = x.double() @ w.double().t()
        CHECKED[key] = {"max_rel_diff": float((y - ref).abs().max() / ref.abs().max().clamp(min=1e-300)),
                        "bf16_identical": bool(torch.equal(bf16_rounded(y), bf16_rounded(ref)))}
        if not CHECKED[key]["bf16_identical"]:
            raise RuntimeError(f"ozaki and float64 round differently at {key}: {CHECKED[key]}")
    return y


def product(x: torch.Tensor, q, n: int | None = None, rows: int | None = None) -> torch.Tensor:
    """x @ W.T exactly, in float64; ``rows``: take the weight that many rows at a time (the head)."""

    if rows is None:
        return _mul(x, exact(q))
    return torch.cat([_mul(x, exact(q, lo, min(n, lo + rows))) for lo in range(0, n, rows)], dim=1)


def selfcheck(shapes=((2048, 16480, 2560), (300, 324, 10240), (1, 2560, 6144))) -> list[dict]:
    """ozaki against float64 cuBLAS on bf16 operands of a wide dynamic range: their float64 results and the bf16 each
    rounds to; returns one row a shape."""

    import time

    torch.backends.cuda.matmul.allow_tf32 = False
    g = torch.Generator(device="cuda").manual_seed(0)
    out = []
    for m, n, k in shapes:
        x = (torch.randn((m, k), generator=g, device="cuda") * torch.exp2(torch.randint(-12, 4, (m, k), generator=g,
                                                                                      device="cuda").float()))
        w = torch.randn((n, k), generator=g, device="cuda") * 0.02
        x, w = x.to(torch.bfloat16), w.to(torch.bfloat16)
        torch.cuda.synchronize()
        t = time.perf_counter()
        a = ozaki(x, w)
        torch.cuda.synchronize()
        t_oz = time.perf_counter() - t
        t = time.perf_counter()
        b = x.double() @ w.double().t()
        torch.cuda.synchronize()
        t_64 = time.perf_counter() - t
        rel = float((a - b).abs().max() / b.abs().max())
        same = bool(torch.equal(bf16_rounded(a), bf16_rounded(b)))
        out.append({"shape": [m, n, k], "max_rel_diff": rel, "bf16_identical": same, "ozaki_s": t_oz, "fp64_s": t_64})
    return out


def install(w) -> None:
    """Every dense prompt GEMM exact (see the module docstring); thor_dense off. Call after moe_reference.install()."""

    from tensorfold.cuda import moe as moe_mod
    from tensorfold.cuda.nvfp4 import thor_dense
    from tensorfold.families.qwen4_exp.cuda import bf16, forward, mtp, nvfp4_moe

    if _INSTALLED:
        return
    thor_dense.set_active(False)
    head = id(w.head)
    original_mm, original_shared = forward._mm, nvfp4_moe._shared
    original_matmul, original_router = bf16.matmul, moe_mod.router

    def put(y: torch.Tensor, out: torch.Tensor | None) -> torch.Tensor:
        if out is None:
            return y
        out.copy_(y)
        return out

    def _mm(x, q, xs, out, b, **kw):
        if not b.prefill:
            return original_mm(x, q, xs, out, b, **kw)
        if id(q) == head:
            y = product(x, q, int(q.n), HEAD_ROWS)
            LAST["head"] = y.detach().clone()
        else:
            y = product(x, q)
        return put(bf16_rounded(y), out)

    def _shared(lin, x, prompt, f32):
        if not prompt:
            return original_shared(lin, x, prompt, f32)
        y = product(x, lin)
        return y.float() if f32 else bf16_rounded(y)

    def matmul(x, b, *, out=None, f32=False, **kw):
        if not f32:                                   # the bf16 faces go through ``_mm``; decode keeps its kernel
            return original_matmul(x, b, out=out, f32=f32, **kw)
        return put(product(x, b).float(), out)

    def router(x, rows, out=None):
        return put(product(x, rows).float(), out)

    forward._mm = _mm
    mtp._mm = _mm
    nvfp4_moe._shared = _shared
    bf16.matmul = matmul
    moe_mod.router = router
    _INSTALLED.update(mm=original_mm, mtp_mm=original_mm, shared=original_shared, matmul=original_matmul,
                      router=original_router)


def uninstall() -> None:
    from tensorfold.cuda import moe as moe_mod
    from tensorfold.families.qwen4_exp.cuda import bf16, forward, mtp, nvfp4_moe

    if _INSTALLED:
        forward._mm = _INSTALLED.pop("mm")
        mtp._mm = _INSTALLED.pop("mtp_mm")
        nvfp4_moe._shared = _INSTALLED.pop("shared")
        bf16.matmul = _INSTALLED.pop("matmul")
        moe_mod.router = _INSTALLED.pop("router")
