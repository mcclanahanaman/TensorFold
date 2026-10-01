// Flash Next's DeltaNet front and back ends, with gdn.cu's chain-kernel operations and lane order, so the same bits.
// Jetson Thor's chunked prompt chain (thor_gdn.py) takes q, k and v in 16 bits and returns y in them: the front writes
// its fp32 q and k, and its bf16 v, rounded once to that type, and the back reads y in it; fp32/bf16 is the default.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int DK = 128, DV = 128, TAPS = 4;

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

template <typename T> __device__ __forceinline__ T store_as(float x);
template <> __device__ __forceinline__ float store_as<float>(float x) { return x; }
template <> __device__ __forceinline__ __nv_bfloat16 store_as<__nv_bfloat16>(float x) { return __float2bfloat16_rn(x); }
template <> __device__ __forceinline__ __half store_as<__half>(float x) { return __float2half_rn(x); }

__device__ __forceinline__ float load_as(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float load_as(__half x) { return __half2float(x); }
__device__ __forceinline__ float load_as(float x) { return x; }

__device__ __forceinline__ float warp_sum(float x) {
    for (int o = 16; o; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float softplusf_(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

// conv + SiLU of channel c for row r: taps from the row's window, [conv state (3 rows) | window rows]
__device__ __forceinline__ float conv_act(const __nv_bfloat16* P, const __nv_bfloat16* cs, const __nv_bfloat16* cw,
                                          const int* win, int r, int c, int C, int PW) {
    float acc = 0.0f;
#pragma unroll
    for (int tap = 0; tap < TAPS; ++tap) {
        const int src = win[r * TAPS + tap];
        const float x = src < TAPS - 1 ? __bfloat162float(cs[src * C + c])
                                       : __bfloat162float(P[(size_t)(src - (TAPS - 1)) * PW + c]);
        acc = acc + __bfloat162float(cw[c * TAPS + tap]) * x;
    }
    return bf(acc / (1.0f + expf(-acc)));
}

// One row of front_kernel's block (row, head): heads < NK L2-norm q (times DK^-0.5) and k as the chain kernel's warps
// 0 and 1; others v, g, beta. QK, V: the stored types of q and k (fp32 by default) and of v (bf16 by default); each
// value is rounded once to it.
template <int NK, int NV, typename QK, typename V>
__device__ __forceinline__ void front_row(
        const __nv_bfloat16* __restrict__ P, const long long* __restrict__ conv_ptrs, const int* __restrict__ sid,
        const int* __restrict__ win, const __nv_bfloat16* __restrict__ cw, const float* __restrict__ a_log,
        const float* __restrict__ dt_bias, QK* __restrict__ q, QK* __restrict__ k,
        V* __restrict__ v, float* __restrict__ g, float* __restrict__ beta, int r) {
    constexpr int C = 2 * NK * DK + NV * DV, PW = C + NV * DV + 2 * NV;
    const int head = blockIdx.y, t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const auto* cs = reinterpret_cast<const __nv_bfloat16*>(conv_ptrs[sid[r]]);
    __shared__ float xs[2][DK];
    if (head < NK) {
        xs[0][t] = conv_act(P, cs, cw, win, r, head * DK + t, C, PW);
        xs[1][t] = conv_act(P, cs, cw, win, r, NK * DK + head * DK + t, C, PW);
        __syncthreads();
        if (warp < 2) {
            float v4[4], ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) { v4[i] = xs[warp][lane * 4 + i]; ss = ss + v4[i] * v4[i]; }
            ss = warp_sum(ss);
            float inv = 1.0f / sqrtf(ss + 1e-6f);
            if (warp == 0) inv = inv * (1.0f / sqrtf((float)DK));
            QK* out = (warp == 0 ? q : k) + ((size_t)r * NK + head) * DK + lane * 4;
#pragma unroll
            for (int i = 0; i < 4; ++i) out[i] = store_as<QK>(v4[i] * inv);
        }
        return;
    }
    const int hv = head - NK;
    v[((size_t)r * NV + hv) * DV + t] = store_as<V>(conv_act(P, cs, cw, win, r, 2 * NK * DK + hv * DV + t, C, PW));
    if (t == 0) {
        const float b = __bfloat162float(P[(size_t)r * PW + C + NV * DV + hv]);
        const float a = __bfloat162float(P[(size_t)r * PW + C + NV * DV + NV + hv]);
        g[r * NV + hv] = expf(-expf(a_log[hv]) * softplusf_(a + dt_bias[hv]));
        beta[r * NV + hv] = bf(sigmoidf_(b));
    }
}

// Block (rows, head): RB rows in turn (RB > 1: fewer, longer blocks for long prompt passes; the same arithmetic).
template <int NK, int NV, typename QK = float, typename V = __nv_bfloat16, int RB = 1>
__global__ void __launch_bounds__(128) front_kernel(
        const __nv_bfloat16* __restrict__ P, const long long* __restrict__ conv_ptrs, const int* __restrict__ sid,
        const int* __restrict__ win, const __nv_bfloat16* __restrict__ cw, const float* __restrict__ a_log,
        const float* __restrict__ dt_bias, QK* __restrict__ q, QK* __restrict__ k,
        V* __restrict__ v, float* __restrict__ g, float* __restrict__ beta, int rows) {
    const int first = blockIdx.x * RB, end = min(rows, first + RB);
    for (int r = first; r < end; ++r) {
        if (r > first) __syncthreads();                          // the last row's norms have read xs
        front_row<NK, NV, QK, V>(P, conv_ptrs, sid, win, cw, a_log, dt_bias, q, k, v, g, beta, r);
    }
}

// One row of back_kernel's block (row, value head): gated RMSNorm on warp 0's sum as the chain kernel, then bf16 out
// and 32-channel group sums. Y: the read-out's stored type (bf16 by default; fp16 from Thor's chunked chain, fp32 from
// a G-B referee's).
template <int NK, int NV, typename Y>
__device__ __forceinline__ void back_row(
        const Y* __restrict__ y, const __nv_bfloat16* __restrict__ P,
        const __nv_bfloat16* __restrict__ norm_w, float eps, __nv_bfloat16* __restrict__ out,
        float* __restrict__ xs, int r) {
    constexpr int C = 2 * NK * DK + NV * DV, PW = C + NV * DV + 2 * NV;
    const int hv = blockIdx.y, t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float ys[DV];
    __shared__ float rinv;
    ys[t] = load_as(y[((size_t)r * NV + hv) * DV + t]);
    __syncthreads();
    if (warp == 0) {
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) { const float yy = ys[lane * 4 + i]; ss = ss + yy * yy; }
        ss = warp_sum(ss);
        if (lane == 0) rinv = 1.0f / sqrtf(ss / (float)DV + eps);
    }
    __syncthreads();
    const float yn = bf(bf(ys[t] * rinv) * __bfloat162float(norm_w[t]));
    const float z = __bfloat162float(P[(size_t)r * PW + C + hv * DV + t]);
    const float o = bf(yn * sigmoidf_(z));
    out[(size_t)r * NV * DV + hv * DV + t] = __float2bfloat16_rn(o);
    const float gs = warp_sum(o);
    if (lane == 0) xs[(size_t)r * (NV * DV / 32) + hv * (DV / 32) + warp] = gs;
}

// Block (rows, value head): RB rows in turn, as front_kernel.
template <int NK, int NV, typename Y = __nv_bfloat16, int RB = 1>
__global__ void __launch_bounds__(128) back_kernel(
        const Y* __restrict__ y, const __nv_bfloat16* __restrict__ P,
        const __nv_bfloat16* __restrict__ norm_w, float eps, __nv_bfloat16* __restrict__ out,
        float* __restrict__ xs, int rows) {
    const int first = blockIdx.x * RB, end = min(rows, first + RB);
    for (int r = first; r < end; ++r) {
        if (r > first) __syncthreads();                          // the last row's sums have read ys and rinv
        back_row<NK, NV, Y>(y, P, norm_w, eps, out, xs, r);
    }
}

}  // namespace

template <typename QK, typename V, int RB>
static void front_launch(const at::Tensor& P, const at::Tensor& conv_ptrs, const at::Tensor& sid, const at::Tensor& win,
                         const at::Tensor& cw, const at::Tensor& a_log, const at::Tensor& dt_bias, at::Tensor& q,
                         at::Tensor& k, at::Tensor& v, at::Tensor& g, at::Tensor& beta) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const int rows = (int)win.size(0), nv = (int)a_log.numel();
    auto launch = [&](auto kernel, int nk) {
        kernel<<<dim3((rows + RB - 1) / RB, nk + nv), 128, 0, stream>>>(
            (const __nv_bfloat16*)P.data_ptr(), (const long long*)conv_ptrs.data_ptr(), sid.data_ptr<int>(),
            win.data_ptr<int>(), (const __nv_bfloat16*)cw.data_ptr(), a_log.data_ptr<float>(),
            dt_bias.data_ptr<float>(), (QK*)q.data_ptr(), (QK*)k.data_ptr(), (V*)v.data_ptr(),
            g.data_ptr<float>(), beta.data_ptr<float>(), rows);
    };
    if (nv == 48) launch(front_kernel<16, 48, QK, V, RB>, 16);
    else if (nv == 24) launch(front_kernel<8, 24, QK, V, RB>, 8);
    else TORCH_CHECK(false, "gdn front: 48 or 24 value heads");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// q, k fp32 and v bf16 (TensorFold's chains), or q, k and v all bf16 or all fp16 (Jetson Thor's chunked chain).
// ``rows_per_block`` 1 (TensorFold's launch) or 8 (Thor's chunked chain: fewer, longer blocks), the same values.
void gdn_front_cuda(const at::Tensor& P, const at::Tensor& conv_ptrs, const at::Tensor& sid, const at::Tensor& win,
                    const at::Tensor& cw, const at::Tensor& a_log, const at::Tensor& dt_bias, at::Tensor& q,
                    at::Tensor& k, at::Tensor& v, at::Tensor& g, at::Tensor& beta, int rows_per_block) {
    const auto qt = q.scalar_type(), vt = v.scalar_type();
    TORCH_CHECK(rows_per_block == 1 || rows_per_block == 8, "gdn front: 1 or 8 rows a block");
    const bool eight = rows_per_block == 8;
    if (qt == at::kFloat && vt == at::kBFloat16) {
        if (eight) front_launch<float, __nv_bfloat16, 8>(P, conv_ptrs, sid, win, cw, a_log, dt_bias, q, k, v, g, beta);
        else front_launch<float, __nv_bfloat16, 1>(P, conv_ptrs, sid, win, cw, a_log, dt_bias, q, k, v, g, beta);
    } else if (qt == at::kBFloat16 && vt == at::kBFloat16) {
        if (eight) front_launch<__nv_bfloat16, __nv_bfloat16, 8>(P, conv_ptrs, sid, win, cw, a_log, dt_bias, q, k, v, g,
                                                                 beta);
        else front_launch<__nv_bfloat16, __nv_bfloat16, 1>(P, conv_ptrs, sid, win, cw, a_log, dt_bias, q, k, v, g,
                                                           beta);
    } else if (qt == at::kHalf && vt == at::kHalf) {
        if (eight) front_launch<__half, __half, 8>(P, conv_ptrs, sid, win, cw, a_log, dt_bias, q, k, v, g, beta);
        else front_launch<__half, __half, 1>(P, conv_ptrs, sid, win, cw, a_log, dt_bias, q, k, v, g, beta);
    } else TORCH_CHECK(false, "gdn front: q, k fp32 with v bf16, or q, k, v all bf16 or all fp16");
}

template <typename Y, int RB>
static void back_launch(const at::Tensor& y, const at::Tensor& P, const at::Tensor& norm_w, double eps, at::Tensor& out,
                        at::Tensor& xs) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const int rows = (int)y.size(0), nv = (int)y.size(1);
    auto launch = [&](auto kernel) {
        kernel<<<dim3((rows + RB - 1) / RB, nv), 128, 0, stream>>>(
            (const Y*)y.data_ptr(), (const __nv_bfloat16*)P.data_ptr(),
            (const __nv_bfloat16*)norm_w.data_ptr(), (float)eps, (__nv_bfloat16*)out.data_ptr(), xs.data_ptr<float>(),
            rows);
    };
    if (nv == 48) launch(back_kernel<16, 48, Y, RB>);
    else if (nv == 24) launch(back_kernel<8, 24, Y, RB>);
    else TORCH_CHECK(false, "gdn back: 48 or 24 value heads");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename Y>
static void back_rows(const at::Tensor& y, const at::Tensor& P, const at::Tensor& norm_w, double eps, at::Tensor& out,
                      at::Tensor& xs, int rows_per_block) {
    if (rows_per_block == 8) back_launch<Y, 8>(y, P, norm_w, eps, out, xs);
    else back_launch<Y, 1>(y, P, norm_w, eps, out, xs);
}

void gdn_back_cuda(const at::Tensor& y, const at::Tensor& P, const at::Tensor& norm_w, double eps, at::Tensor& out,
                   at::Tensor& xs, int rows_per_block) {
    TORCH_CHECK(rows_per_block == 1 || rows_per_block == 8, "gdn back: 1 or 8 rows a block");
    if (y.scalar_type() == at::kBFloat16) back_rows<__nv_bfloat16>(y, P, norm_w, eps, out, xs, rows_per_block);
    else if (y.scalar_type() == at::kHalf) back_rows<__half>(y, P, norm_w, eps, out, xs, rows_per_block);
    else if (y.scalar_type() == at::kFloat) back_rows<float>(y, P, norm_w, eps, out, xs, rows_per_block);
    else TORCH_CHECK(false, "gdn back: y bf16, fp16 or fp32");
}
