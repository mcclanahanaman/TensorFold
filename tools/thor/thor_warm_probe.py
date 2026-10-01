"""Where the Thor experts' startup time and memory go: compile time per warm call, CuTe DSL's file cache, and the
process's resident and GPU-visible memory around it (Jetson's memory is unified: host allocations shrink what the
caches can take).

    CUTE_DSL_CACHE_DIR=/cache/cute_dsl python tools/thor/thor_warm_probe.py --out /out/warm_probe.json

Builds one layer's experts at Flash Next's shapes (random bytes; compile time and memory do not depend on values),
then warms a 28-row decode scratch and a 2048-row prompt scratch, timing each synthetic call.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import torch


def mem() -> dict:
    free, total = torch.cuda.mem_get_info()
    rss = 0
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
    except OSError:
        pass
    return {"gpu_free_gib": free / 2**30, "rss_gib": rss / 2**30,
            "torch_reserved_gib": torch.cuda.memory_reserved() / 2**30,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/out/warm_probe.json")
    args = ap.parse_args()
    rep: dict = {"env": {k: os.environ.get(k) for k in ("CUTE_DSL_CACHE_DIR", "CUTE_DSL_DISABLE_FILE_CACHING")}}
    rep["mem_start"] = mem()
    t0 = time.time()
    from tensorfold.cuda.nvfp4 import thor_experts as thor

    thor._fi()
    rep["import_s"] = time.time() - t0
    try:
        import cutlass.base_dsl.cache_helpers as ch
        from cutlass.cutlass_dsl import CuTeDSL

        dsl = CuTeDSL._get_dsl()
        rep["dsl"] = {"cache_dir": str(ch.default_generated_ir_path),
                      "disable_file_caching": bool(getattr(dsl.envar, "disable_file_caching", None)),
                      "arch": str(getattr(dsl.envar, "arch", None))}
    except Exception as exc:  # noqa: BLE001
        rep["dsl"] = repr(exc)
    rep["mem_after_import"] = mem()
    e, ni, d = 512, 640, 2560
    g = torch.Generator(device="cpu").manual_seed(0)

    def proj(n, k):
        return (torch.randint(0, 256, (e, n, k // 2), generator=g, dtype=torch.uint8).cuda(),
                torch.randint(0x28, 0x40, (e, n, k // 16), generator=g, dtype=torch.uint8).cuda(),
                torch.full((e,), 0.01).cuda())

    s2 = torch.full((e,), 0.01).cuda()
    gate, up, down = proj(ni, d), proj(ni, d), proj(d, ni)
    up = (up[0], up[1], s2)
    gate = (gate[0], gate[1], s2)
    ex = thor.make(gate, up, down)
    del gate, up, down
    torch.cuda.empty_cache()
    rep["mem_after_layout"] = mem()
    calls = []
    original = thor.run

    def timed_run(x, ex_, plan, y, rows, top_k, scr):
        torch.cuda.synchronize()
        t = time.time()
        original(x, ex_, plan, y, rows, top_k, scr)
        torch.cuda.synchronize()
        calls.append({"rows": rows, "slots": str(y.dtype), "seconds": time.time() - t})

    thor.run = timed_run
    for rows, prompt in ((28, True), (28, False), (2048, True)):
        t = time.time()
        scr = thor.Scratch(rows, 10, e, ni, d, "cuda")
        thor.warm(ex, scr, 10, prompt=prompt)
        rep[f"warm_{rows}_{'prompt' if prompt else 'decode'}_s"] = time.time() - t
        rep[f"mem_after_warm_{rows}_{prompt}"] = mem()
    thor.run = original
    rep["calls"] = calls
    cache = Path(rep["dsl"]["cache_dir"]) if isinstance(rep["dsl"], dict) else None
    if cache is not None and cache.exists():
        files = list(cache.rglob("*"))
        rep["cache_files"] = len(files)
        rep["cache_mb"] = sum(f.stat().st_size for f in files if f.is_file()) / 2**20
    Path(args.out).write_text(json.dumps(rep, indent=1, default=str))
    for c in calls:
        print(f"warm call rows={c['rows']} slots={c['slots']}: {c['seconds']:.2f}s", flush=True)
    print(json.dumps({k: v for k, v in rep.items() if k != "calls"}, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
