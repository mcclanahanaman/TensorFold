// Qwen's vision-tower rotary embedding on Jetson Thor (sm_110) (THOR-VISION-ATTN): transformers'
// apply_rotary_pos_emb_vision, bit for bit, in one pass over the qkv GEMM's rows.
//
// transformers casts q and k to fp32, rounds q * cos and rotate_half(q) * sin as two fp32 products, adds them with one
// rounding and casts the sum to bf16 (round to nearest even): about seven full-size fp32 passes a tensor, 45% of a
// video's tower time on Thor. Here one thread takes two neighbouring elements of a head's first half and their
// partners in the second half and does the same operations in the same order: __fmul_rn and __fadd_rn (no fused
// multiply-add contraction), the negation before the product as rotate_half does, and __float2bfloat16 (the device
// conversion c10::BFloat16 itself uses). bf16 -> fp32 is exact, and so is reading cos and sin as stored (fp32).
// q and k are read straight from the qkv rows and written contiguous, [rows, heads, head_dim], for the attention.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

__global__ void rope_kernel(const __nv_bfloat16* __restrict__ qkv, int64_t row_stride, int64_t k_offset,
                            const float* __restrict__ cos, const float* __restrict__ sin, int64_t cos_stride,
                            int64_t sin_stride, __nv_bfloat16* __restrict__ q_out, __nv_bfloat16* __restrict__ k_out,
                            int64_t rows, int heads, int half) {
  const int pairs = half / 2;                       // element pairs in a head's first half
  const int64_t per_row = 2LL * heads * pairs;      // q and k
  const int64_t total = rows * per_row;
  const int dim = 2 * half;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < total; i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t r = i / per_row;
    int rest = (int)(i - r * per_row);
    const int which = rest / (heads * pairs);        // 0: q, 1: k
    rest -= which * heads * pairs;
    const int h = rest / pairs;
    const int d = 2 * (rest - h * pairs);            // first-half element; its partner is d + half
    const __nv_bfloat16* src = qkv + r * row_stride + which * k_offset + (int64_t)h * dim;
    const __nv_bfloat162 x1 = *reinterpret_cast<const __nv_bfloat162*>(src + d);
    const __nv_bfloat162 x2 = *reinterpret_cast<const __nv_bfloat162*>(src + half + d);
    const float2 c1 = *reinterpret_cast<const float2*>(cos + r * cos_stride + d);
    const float2 c2 = *reinterpret_cast<const float2*>(cos + r * cos_stride + half + d);
    const float2 s1 = *reinterpret_cast<const float2*>(sin + r * sin_stride + d);
    const float2 s2 = *reinterpret_cast<const float2*>(sin + r * sin_stride + half + d);
    const float a0 = __bfloat162float(x1.x), a1 = __bfloat162float(x1.y);
    const float b0 = __bfloat162float(x2.x), b1 = __bfloat162float(x2.y);
    // rotate_half(x) = cat(-x2, x1): out[d] = x1 * cos[d] + (-x2) * sin[d],
    //                                 out[half + d] = x2 * cos[half + d] + x1 * sin[half + d]
    __nv_bfloat162 lo, hi;
    lo.x = __float2bfloat16(__fadd_rn(__fmul_rn(a0, c1.x), __fmul_rn(-b0, s1.x)));
    lo.y = __float2bfloat16(__fadd_rn(__fmul_rn(a1, c1.y), __fmul_rn(-b1, s1.y)));
    hi.x = __float2bfloat16(__fadd_rn(__fmul_rn(b0, c2.x), __fmul_rn(a0, s2.x)));
    hi.y = __float2bfloat16(__fadd_rn(__fmul_rn(b1, c2.y), __fmul_rn(a1, s2.y)));
    __nv_bfloat16* dst = (which ? k_out : q_out) + (r * heads + h) * (int64_t)dim;
    *reinterpret_cast<__nv_bfloat162*>(dst + d) = lo;
    *reinterpret_cast<__nv_bfloat162*>(dst + half + d) = hi;
  }
}

}  // namespace

void rope_launch(const at::Tensor& qkv, const at::Tensor& cos, const at::Tensor& sin, at::Tensor& q, at::Tensor& k,
                 int64_t heads, int64_t head_dim) {
  const int64_t rows = qkv.size(0);
  const int64_t total = rows * heads * head_dim / 2;   // threads: (q, k) x heads x head_dim / 4 pairs a row
  if (total == 0) return;
  const int threads = 256;
  const int64_t blocks = std::min<int64_t>((total + threads - 1) / threads, int64_t(1) << 22);
  rope_kernel<<<(unsigned)blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr<at::BFloat16>()), qkv.stride(0), heads * head_dim,
      cos.data_ptr<float>(), sin.data_ptr<float>(), cos.stride(0), sin.stride(0),
      reinterpret_cast<__nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(k.data_ptr<at::BFloat16>()), rows, (int)heads, (int)(head_dim / 2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
