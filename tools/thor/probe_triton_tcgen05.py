"""Does this container's Triton lower a bf16 tl.dot to tcgen05 (UMMA) on sm_110, and how fast is it?

    python tools/thor/probe_triton_tcgen05.py > /out/triton_tcgen05.txt

Prints the Triton version, the target, which MMA instructions the PTX of a plain bf16 matmul contains (tcgen05.mma,
wgmma, mma.sync), and its TFLOPS at a few shapes (CUDA events, warm-up excluded) next to cuBLAS bf16.
"""

from __future__ import annotations

import re

import torch
import triton
import triton.language as tl


@triton.jit
def _mm(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    npm, npn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    group = pid // (GROUP * npn)
    first = group * GROUP
    size = min(npm - first, GROUP)
    pm = first + (pid % size)
    pn = (pid % (GROUP * npn)) // size
    a_desc = tl.make_tensor_descriptor(A, shape=[M, K], strides=[K, 1], block_shape=[BM, BK])
    b_desc = tl.make_tensor_descriptor(B, shape=[N, K], strides=[K, 1], block_shape=[BN, BK])
    acc = tl.zeros((BM, BN), tl.float32)
    for k in tl.range(0, K, BK, num_stages=3):
        a = a_desc.load([pm * BM, k])
        b = b_desc.load([pn * BN, k])
        acc = tl.dot(a, b.T, acc)
    c_desc = tl.make_tensor_descriptor(C, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    c_desc.store([pm * BM, pn * BN], acc.to(tl.bfloat16))


@triton.jit
def _mm_ptr(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        a = tl.load(A + rm[:, None] * K + (k + rk)[None, :])
        b = tl.load(B + rn[:, None] * K + (k + rk)[None, :])
        acc = tl.dot(a, b.T, acc)
    tl.store(C + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16))


def ev(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / reps


def kinds(asm: str) -> dict:
    return {k: len(re.findall(k, asm)) for k in ("tcgen05.mma", "tcgen05.ld", "wgmma.mma_async", "mma.sync",
                                                    "cp.async.bulk.tensor")}


def main():
    print("triton", triton.__version__, "torch", torch.__version__, "device",
          torch.cuda.get_device_name(), torch.cuda.get_device_capability())
    try:
        print("target", triton.runtime.driver.active.get_current_target())
    except Exception as exc:  # noqa: BLE001
        print("target ?", exc)

    def alloc(size, align, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc)
    for (m, n, k) in ((4096, 4096, 4096), (2048, 1280, 2560), (8192, 2560, 640)):
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        c = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        ref = (a.float() @ b.float().t())
        fl = 2.0 * m * n * k
        t = ev(lambda: torch.mm(a, b.t()))
        print(f"  cuBLAS bf16 {m}x{n}x{k}: {t:.3f} ms {fl / t / 1e9:.1f} TFLOPS")
        for name, launch in (
            ("desc 128x128x64 w4", lambda: _mm[(triton.cdiv(m, 128) * triton.cdiv(n, 128),)](
                a, b, c, m, n, k, BM=128, BN=128, BK=64, GROUP=8, num_warps=4)),
            ("desc 128x256x64 w8", lambda: _mm[(triton.cdiv(m, 128) * triton.cdiv(n, 256),)](
                a, b, c, m, n, k, BM=128, BN=256, BK=64, GROUP=8, num_warps=8)),
            ("ptr 128x128x64 w4 s3", lambda: _mm_ptr[(triton.cdiv(m, 128), triton.cdiv(n, 128))](
                a, b, c, m, n, k, BM=128, BN=128, BK=64, num_warps=4, num_stages=3)),
        ):
            try:
                kern = launch()
                torch.cuda.synchronize()
                err = float((c.float() - ref).norm() / ref.norm())
                t = ev(launch)
                print(f"  triton {name} {m}x{n}x{k}: {t:.3f} ms {fl / t / 1e9:.1f} TFLOPS rel {err:.2e} "
                      f"ptx {kinds(kern.asm['ptx'])}")
            except Exception as exc:  # noqa: BLE001
                print(f"  triton {name}: FAILED {type(exc).__name__}: {str(exc)[:600]}")
    print("dot_scaled", hasattr(tl, "dot_scaled"))


if __name__ == "__main__":
    main()
