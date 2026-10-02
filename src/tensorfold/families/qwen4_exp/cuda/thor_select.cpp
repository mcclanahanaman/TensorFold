#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void thor_select_cuda(const at::Tensor&, const at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, int64_t, int64_t,
                      int64_t, int64_t, int64_t);

static void check(const at::Tensor& x, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kInt, name, ": expected an int32 CUDA tensor");
}

// attention._select's lists for rows 0 .. rows - 1 of a prompt block (thor_select.cu): scores fp32 [>= rows, >= blocks]
// with unit column stride; ids int32 [>= rows, budget + ratio]; nk, sparse int32 [>= rows]; pos0 the block's first
// position (on the device).
void select_lists(const at::Tensor& scores, const at::Tensor& pos0, at::Tensor ids, at::Tensor nk, at::Tensor sparse,
                  int64_t rows, int64_t blocks, int64_t ratio, int64_t top, int64_t variant) {
    TORCH_CHECK(scores.is_cuda() && scores.scalar_type() == at::kFloat && scores.dim() == 2 && scores.stride(1) == 1,
                "scores: an fp32 CUDA matrix with unit column stride");
    check(pos0, "pos0");
    check(ids, "ids");
    check(nk, "nk");
    check(sparse, "sparse");
    TORCH_CHECK(pos0.numel() >= 1 && ids.dim() == 2 && ids.stride(1) == 1 && nk.is_contiguous() &&
                sparse.is_contiguous(), "ids a matrix with unit column stride; nk and sparse contiguous");
    TORCH_CHECK(rows >= 0 && scores.size(0) >= rows && ids.size(0) >= rows && nk.numel() >= rows &&
                sparse.numel() >= rows && blocks >= 1 && scores.size(1) >= blocks,
                "rows and blocks within the buffers");
    TORCH_CHECK(ratio >= 1 && top >= 1 && ids.size(1) >= top * ratio + ratio - 1, "a list holds top blocks and a tail");
    c10::cuda::CUDAGuard guard(scores.device());
    thor_select_cuda(scores, pos0, ids, nk, sparse, rows, blocks, ratio, top, variant);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("select", &select_lists); }
