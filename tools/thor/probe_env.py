"""What the Thor dev image offers for NVFP4 experts: the GPU, torch, FlashInfer's MoE entry points, CUTLASS, CuTe DSL.

Run inside the runner's container (nothing here writes outside /out):

    python tools/thor/probe_env.py /hf/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/<rev> > /out/env.txt
"""

from __future__ import annotations

import glob
import importlib
import importlib.metadata as md
import json
import os
import struct
import subprocess
import sys
from pathlib import Path


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as exc:  # noqa: BLE001 - a probe reports, it never stops
        return f"<{exc!r}>"


def section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def gpu() -> None:
    import torch

    section("torch / GPU")
    print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
    p = torch.cuda.get_device_properties(0)
    print("device", p.name, "cc", (p.major, p.minor), "SMs", p.multi_processor_count, "mem GiB",
          round(p.total_memory / 2 ** 30, 1))
    for attr in ("shared_memory_per_block_optin", "shared_memory_per_multiprocessor", "L2_cache_size",
                 "max_threads_per_multi_processor", "regs_per_multiprocessor", "clock_rate", "memory_clock_rate",
                 "memory_bus_width"):
        print(" ", attr, getattr(p, attr, "n/a"))
    print("arch list", torch.cuda.get_arch_list())
    print("float4_e2m1fn_x2", hasattr(torch, "float4_e2m1fn_x2"), "_scaled_mm", hasattr(torch, "_scaled_mm"),
          "_grouped_mm", hasattr(torch, "_grouped_mm"), "_scaled_grouped_mm", hasattr(torch, "_scaled_grouped_mm"))
    print("nvcc:", sh("which nvcc; nvcc --version | tail -2"))
    print("cpu:", sh("nproc"), "cores;", sh("free -g | head -2 | tail -1"))
    print("TORCH_CUDA_ARCH_LIST", os.environ.get("TORCH_CUDA_ARCH_LIST"),
          "FLASHINFER_CUDA_ARCH_LIST", os.environ.get("FLASHINFER_CUDA_ARCH_LIST"))


def packages() -> None:
    section("packages")
    want = ("flashinfer", "flashinfer-python", "flashinfer-jit-cache", "flashinfer-cubin", "nvidia-cutlass-dsl",
            "nvidia-cutlass", "cutlass", "triton", "nvidia-cublas", "nvidia-cublas-cu13", "nvidia-cublas-cu12",
            "cuda-python", "cuda-bindings", "tensorrt_llm", "vllm", "sgl-kernel", "deep_gemm", "safetensors",
            "numpy", "ninja")
    for d in sorted(md.distributions(), key=lambda d: d.metadata["Name"].lower()):
        name = d.metadata["Name"]
        if any(w in name.lower() for w in ("flashinfer", "cutlass", "cublas", "cuda", "triton", "nvidia",
                                           "tensorrt", "deep", "safetensors", "numpy", "ninja", "torch")) \
                or name in want:
            print(f"  {name}=={d.version}")


def flashinfer() -> None:
    section("flashinfer")
    try:
        import flashinfer
    except Exception as exc:  # noqa: BLE001
        print("import failed", repr(exc))
        return
    root = Path(flashinfer.__file__).parent
    print("version", getattr(flashinfer, "__version__", "?"), "at", root)
    try:
        from flashinfer import fused_moe

        names = sorted(n for n in dir(fused_moe) if not n.startswith("_"))
        print("fused_moe exports:", names)
    except Exception as exc:  # noqa: BLE001
        print("fused_moe import failed", repr(exc))
    for mod in ("flashinfer.cute_dsl", "flashinfer.gemm", "flashinfer.fp4_quantization", "flashinfer.jit",
                "flashinfer.jit.env", "flashinfer.jit.core", "flashinfer.compilation_context"):
        try:
            m = importlib.import_module(mod)
            print(mod, "->", sorted(n for n in dir(m) if not n.startswith("_"))[:120])
        except Exception as exc:  # noqa: BLE001
            print(mod, "import failed", repr(exc))
    try:
        from flashinfer.jit import env

        for k in sorted(dir(env)):
            if k.isupper():
                print("  jit.env", k, "=", getattr(env, k))
    except Exception as exc:  # noqa: BLE001
        print("jit.env failed", repr(exc))
    print("fused_moe sources:", sh(f"ls {root}/fused_moe 2>/dev/null; ls {root}/data 2>/dev/null | head -50"))
    print("csrc moe:", sh(f"find {root} -path '*fused_moe*' -maxdepth 6 | head -60"))
    for pat in ("flashinfer_jit_cache", "flashinfer_cubin"):
        try:
            m = importlib.import_module(pat)
            base = Path(m.__file__).parent
            print(pat, "at", base)
            print(sh(f"find {base} -maxdepth 4 | grep -i -E 'moe|fp4|sm11|sm_11|110' | head -80"))
            print(sh(f"du -sh {base}; find {base} -maxdepth 3 -type d | head -40"))
        except Exception as exc:  # noqa: BLE001
            print(pat, "not importable", repr(exc))


def cutlass() -> None:
    section("CUTLASS headers")
    hits = sorted(set(glob.glob("/usr/**/cutlass/version.h", recursive=True)
                      + glob.glob(sys.prefix + "/lib/python3*/site-packages/**/cutlass/version.h", recursive=True)
                      + glob.glob("/opt/**/cutlass/version.h", recursive=True)))
    for h in hits[:12]:
        text = Path(h).read_text()
        ver = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("#define CUTLASS_") and "VERSION" in ln
               or ln.strip().startswith("#define CUTLASS_MAJOR") or ln.strip().startswith("#define CUTLASS_MINOR")
               or ln.strip().startswith("#define CUTLASS_PATCH")]
        print(h, ver[:6])
        inc = Path(h).parent.parent
        print("   sm100 collectives:", sh(f"ls {inc}/cutlass/gemm/collective/ | grep -E 'sm100|sm110|mixed' | head -40"))
        print("   SM110 mentions:", sh(f"grep -rl -E 'SM110|sm110|Sm110' {inc} 2>/dev/null | head -20"))
    section("CuTe DSL")
    try:
        import cutlass as cdsl  # noqa: F401

        print("cutlass python at", Path(cdsl.__file__).parent, getattr(cdsl, "__version__", "?"))
        base = Path(cdsl.__file__).parent
        print(sh(f"grep -rn -E 'sm_110|sm_101|110a|101a' {base} --include=*.py 2>/dev/null | head -30"))
    except Exception as exc:  # noqa: BLE001
        print("cutlass DSL import failed", repr(exc))


def header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def checkpoint(model: str) -> None:
    section("checkpoint")
    root = Path(model)
    print(sh(f"ls -la {root} | head -40"))
    cfg = json.loads((root / "config.json").read_text())
    q = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config")
    print("quantization_config:", json.dumps(q)[:3000])
    idx = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    names = sorted(idx)
    print("tensors", len(names))
    for layer in (0, 23, 47):
        sel = [n for n in names if f"layers.{layer}.mlp" in n and (".experts.0." in n or ".experts.511." in n
                                                                      or "experts." not in n)]
        print(f"layer {layer} mlp tensors (expert 0/511 + non-expert):")
        for n in sel[:60]:
            h = header(root / idx[n])
            print("   ", n, h[n]["dtype"], h[n]["shape"], idx[n])
    scales = [n for n in names if n.endswith("input_scale")]
    print("input_scale tensors:", len(scales), scales[:4])
    h = header(root / idx[names[0]])
    print("metadata:", h.get("__metadata__"))


def main() -> None:
    gpu()
    packages()
    flashinfer()
    cutlass()
    if len(sys.argv) > 1:
        checkpoint(sys.argv[1])


if __name__ == "__main__":
    main()
