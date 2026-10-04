#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void rope_launch(const at::Tensor& qkv, const at::Tensor& cos, const at::Tensor& sin, at::Tensor& q, at::Tensor& k,
                 int64_t heads, int64_t head_dim);

// q and k of a vision attention layer, rotated: qkv [rows, 3 * heads * head_dim] bf16 (rows may be strided, elements
// contiguous), cos and sin [rows, head_dim] fp32 -> q, k [rows, heads, head_dim] bf16, contiguous.
std::vector<at::Tensor> rope(const at::Tensor& qkv, const at::Tensor& cos, const at::Tensor& sin, int64_t heads,
                             int64_t head_dim) {
  TORCH_CHECK(qkv.is_cuda() && qkv.scalar_type() == at::kBFloat16 && qkv.dim() == 2 && qkv.stride(1) == 1,
              "qkv must be a CUDA bf16 matrix with contiguous rows");
  TORCH_CHECK(heads > 0 && head_dim > 0 && head_dim % 4 == 0 && qkv.size(1) == 3 * heads * head_dim,
              "qkv must hold q, k and v of heads x head_dim (head_dim a multiple of 4)");
  const int64_t rows = qkv.size(0);
  for (const auto* t : {&cos, &sin}) {
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kFloat && t->dim() == 2 && t->size(0) == rows &&
                t->size(1) == head_dim && t->stride(1) == 1 && t->stride(0) % 2 == 0 &&
                reinterpret_cast<uintptr_t>(t->data_ptr()) % 8 == 0,
                "cos and sin must be fp32 [rows, head_dim] with contiguous, 8-byte aligned rows");
  }
  TORCH_CHECK(qkv.stride(0) % 2 == 0 && reinterpret_cast<uintptr_t>(qkv.data_ptr()) % 4 == 0,
              "qkv rows must be 4-byte aligned");
  const c10::cuda::CUDAGuard guard(qkv.device());
  auto q = at::empty({rows, heads, head_dim}, qkv.options());
  auto k = at::empty({rows, heads, head_dim}, qkv.options());
  rope_launch(qkv, cos, sin, q, k, heads, head_dim);
  return {q, k};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rope", &rope, "Qwen vision rotary embedding of q and k, transformers' bits (THOR-VISION-ATTN)");
}
