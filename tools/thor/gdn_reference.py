"""The gated delta rule in fp64, one serial step a row: the G-A reference for Flash Next's DeltaNet prompt chain
(THOR-K4).

``chain64(q, k, v, g, beta, s0)`` runs TensorFold's chain rule -- s *= g; delta = (v - s k) beta; s += delta k^T;
y = s q -- with every input taken exactly (q, k fp32 or 16-bit, v bf16 or 16-bit, g and beta fp32, the state fp32 or
fp64) and every operation in fp64, one CUDA launch for the whole sequence. It returns y (fp64, [T, Hv, 128]) and the
last state (fp64, [Hv, 128, 128], value rows by key columns as TensorFold keeps it). ``readout64(s, q)`` reads a
state out for given queries (the first decode step's y from a prompt's last state).
"""

from __future__ import annotations

import functools

import torch

_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

__device__ __forceinline__ double d(float x) { return (double)x; }
__device__ __forceinline__ double d(__nv_bfloat16 x) { return (double)__bfloat162float(x); }
__device__ __forceinline__ double d(__half x) { return (double)__half2float(x); }

// Block: one value head, 32 value rows; 4 threads a row, 32 key columns each (state in fp64 registers).
template <typename QK, typename V>
__global__ void __launch_bounds__(128) chain64_kernel(const QK* __restrict__ q, const QK* __restrict__ k,
        const V* __restrict__ v, const float* __restrict__ g, const float* __restrict__ beta,
        const double* __restrict__ s0, double* __restrict__ s1, double* __restrict__ y, int W, int hk, int hv) {
    __shared__ double ks[128], qs[128];
    const int head = blockIdx.x, row = blockIdx.y * 32 + (threadIdx.x >> 2), part = threadIdx.x & 3;
    const int kh = head / (hv / hk);
    double s[32];
    const double* src = s0 + ((size_t)head * 128 + row) * 128 + part * 32;
#pragma unroll
    for (int j = 0; j < 32; ++j) s[j] = src[j];
    for (int t = 0; t < W; ++t) {
        __syncthreads();
        ks[threadIdx.x] = d(k[((size_t)t * hk + kh) * 128 + threadIdx.x]);
        qs[threadIdx.x] = d(q[((size_t)t * hk + kh) * 128 + threadIdx.x]);
        __syncthreads();
        const double gt = (double)g[(size_t)t * hv + head], bt = (double)beta[(size_t)t * hv + head];
        double m = 0.0;
#pragma unroll
        for (int j = 0; j < 32; ++j) { s[j] *= gt; m += s[j] * ks[part * 32 + j]; }
        m += __shfl_xor_sync(0xffffffffu, m, 1);
        m += __shfl_xor_sync(0xffffffffu, m, 2);
        const double delta = (d(v[((size_t)t * hv + head) * 128 + row]) - m) * bt;
        double o = 0.0;
#pragma unroll
        for (int j = 0; j < 32; ++j) { s[j] += ks[part * 32 + j] * delta; o += s[j] * qs[part * 32 + j]; }
        o += __shfl_xor_sync(0xffffffffu, o, 1);
        o += __shfl_xor_sync(0xffffffffu, o, 2);
        if (part == 0) y[((size_t)t * hv + head) * 128 + row] = o;
    }
    double* dst = s1 + ((size_t)head * 128 + row) * 128 + part * 32;
#pragma unroll
    for (int j = 0; j < 32; ++j) dst[j] = s[j];
}

template <typename QK, typename V>
static void launch(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g,
                   const at::Tensor& beta, const at::Tensor& s0, at::Tensor& s1, at::Tensor& y) {
    const int W = q.size(0), hk = q.size(1), hv = v.size(1);
    chain64_kernel<QK, V><<<dim3(hv, 4), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const QK*)q.data_ptr(), (const QK*)k.data_ptr(), (const V*)v.data_ptr(), g.data_ptr<float>(),
        beta.data_ptr<float>(), s0.data_ptr<double>(), s1.data_ptr<double>(), y.data_ptr<double>(), W, hk, hv);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<at::Tensor> chain64(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g,
                                const at::Tensor& beta, const at::Tensor& s0) {
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && g.is_contiguous() &&
                beta.is_contiguous() && s0.is_contiguous(), "contiguous inputs");
    TORCH_CHECK(q.dim() == 3 && q.size(2) == 128 && k.sizes() == q.sizes() && v.dim() == 3 && v.size(2) == 128 &&
                v.size(0) == q.size(0) && v.size(1) % q.size(1) == 0, "q, k [T, Hk, 128]; v [T, Hv, 128]");
    TORCH_CHECK(g.scalar_type() == at::kFloat && beta.scalar_type() == at::kFloat && s0.scalar_type() == at::kDouble,
                "g, beta fp32; the state fp64");
    c10::cuda::CUDAGuard guard(q.device());
    auto y = at::empty({q.size(0), v.size(1), 128}, q.options().dtype(at::kDouble));
    auto s1 = at::empty_like(s0);
    if (q.size(0) == 0) { s1.copy_(s0); return {y, s1}; }
    const auto qt = q.scalar_type(), vt = v.scalar_type();
    TORCH_CHECK(k.scalar_type() == qt, "q and k share a dtype");
    if (qt == at::kFloat && vt == at::kBFloat16) launch<float, __nv_bfloat16>(q, k, v, g, beta, s0, s1, y);
    else if (qt == at::kFloat && vt == at::kFloat) launch<float, float>(q, k, v, g, beta, s0, s1, y);
    else if (qt == at::kBFloat16 && vt == at::kBFloat16)
        launch<__nv_bfloat16, __nv_bfloat16>(q, k, v, g, beta, s0, s1, y);
    else if (qt == at::kHalf && vt == at::kHalf) launch<__half, __half>(q, k, v, g, beta, s0, s1, y);
    else TORCH_CHECK(false, "q/k fp32 with v bf16 or fp32, or q, k, v all bf16 or all fp16");
    return {y, s1};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("chain64", &chain64); }
"""


@functools.lru_cache(maxsize=1)
def _ext():
    from torch.utils.cpp_extension import load_inline

    return load_inline(name="tf_thor_gdn_ref64_v1", cpp_sources=[], cuda_sources=[_SRC], functions=None,
                       extra_cuda_cflags=["-O3"], verbose=False)


def chain64(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
            s0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp64 serial chain from ``s0`` ([Hv, 128, 128], any float dtype): (y [T, Hv, 128], last state), both fp64."""

    y, s1 = _ext().chain64(q.contiguous(), k.contiguous(), v.contiguous(), g.float().contiguous(),
                           beta.float().contiguous(), s0.double().contiguous())
    return y, s1


def readout64(s: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """y = s q in fp64 for queries ``q`` [N, Hk, 128] against a state [Hv, 128, 128]: [N, Hv, 128]."""

    hv, hk = s.shape[0], q.shape[1]
    qq = q.double().repeat_interleave(hv // hk, dim=1)
    return torch.einsum("hvk,nhk->nhv", s.double(), qq)


def rel(a: torch.Tensor, ref: torch.Tensor) -> float:
    """Relative Frobenius error of ``a`` against ``ref``."""

    return float((a.double() - ref).norm() / ref.norm().clamp_min(1e-300))


def install_prompt_chain() -> None:
    """A G-B referee's DeltaNet (THOR-K4): every prompt pass's chain exact -- this module's fp64 rule on TensorFold's
    front outputs (q, k fp32; v bf16), the read-out rounded once to fp32 (the back reads it as fp32) and the last state
    once to fp32 (its buffer) -- so neither TensorFold's fp32 chain (bf16 read-outs) nor Thor's chunked one (fp16) is
    the referee's. Decode keeps TensorFold's kernels, which both runs share."""

    from tensorfold.cuda.kernels import gdn as shared
    from tensorfold.cuda.kernels import thor_gdn

    thor_gdn.set_active(False)                       # TensorFold's front and kept points; the chain is replaced below

    def exact(q, k, v, g, beta, state, final):
        y, s = chain64(q, k, v, g, beta, state)
        final.copy_(s)
        return y.float()

    shared.chain = exact


def selfcheck(model: str, tokens: int = 6000) -> dict:
    """One prompt's head logits on TensorFold's chain, on Thor's chunked chain and on this referee's exact chain (the
    same engine, nothing kept between them): each path's distance from the referee.

        python tools/thor/gdn_reference.py MODEL_DIR [TOKENS]
    """

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    import ngram_residency as nr
    import prefill_profile as pp

    from tensorfold.cuda.kernels import gdn as shared
    from tensorfold.cuda.kernels import thor_gdn
    from tensorfold.families.qwen4_exp.cuda import multi as multi_mod

    engine = pp.build_engine(Path(model), 4, "int8")
    ids = pp.prompt_ids(Path(model), tokens, 31337)
    heads: dict = {}
    joined = multi_mod.MultiDecoder._joined

    def catch(dec, pieces, h, lasts, spent):
        if h is not None:
            heads["last"] = h.float().clone()
        return joined(dec, pieces, h, lasts, spent)

    multi_mod.MultiDecoder._joined = catch
    original = shared.chain
    out: dict = {}
    try:
        for name in ("stock", "thor", "referee"):
            nr.forget(engine)
            thor_gdn.set_active(name == "thor")
            if name == "referee":
                install_prompt_chain()
            got = pp.run_prompt(engine, ids)
            out[name] = {"first": got["first"], "logits": heads.pop("last")}
    finally:
        shared.chain = original
        multi_mod.MultiDecoder._joined = joined
        thor_gdn.set_active(thor_gdn.S.ready)
    ref = out["referee"]["logits"]
    report = {"tokens": len(ids), "first": {k: v["first"] for k, v in out.items()}}
    for name in ("stock", "thor"):
        d = out[name]["logits"] - ref
        report[name] = {"max_abs": float(d.abs().max()), "rms": float(d.pow(2).mean().sqrt()),
                        "rel": float(d.norm() / ref.norm())}
    print(f"[gdn-ref] {len(ids)} tokens: first tokens {report['first']}; head logits vs the referee: stock rms "
          f"{report['stock']['rms']:.4f} (max {report['stock']['max_abs']:.3f}), thor rms {report['thor']['rms']:.4f} "
          f"(max {report['thor']['max_abs']:.3f})", flush=True)
    return report


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(selfcheck(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6000), indent=1))
