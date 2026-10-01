"""G-A for Jetson Thor's prompt GEMMs (THOR-K2): real weights, real activations, at M = 128, 512, 2048 and 4096.

    python tools/thor/thor_dense_ga.py MODEL_DIR OUT.json [--layers 0,23,47] [--parallel 4] [--kv-dtype int8]

The engine is built as served (``tensorfold serve MODEL --backend cuda --kv-dtype int8 --parallel 4``, thor_dense on);
one natural-text prompt of 4096 tokens (two prompt passes) runs through it and the input rows of each chosen linear
are kept as they ran: the DeltaNet and attention projections, the shared expert and both hyper-connections of the
layers of each kind nearest ``--layers``, the n-gram key/value and the MTP head's fc. Per cell, against x @ W.T in
float64 on the exact weights: TensorFold's prompt GEMM, thor_dense dequantizing at each call and with a bf16 copy
(their bits must agree), the error ratio to TensorFold's (G-A: <= 1.5), and whether the cell's rows equal the same rows
of the 4096-row call (row invariance).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

M_CELLS = (128, 512, 2048, 4096)


def chosen(w, targets: list[int]) -> dict:
    """id(linear) -> (name, linear, fp32 sums) for the layers of each kind nearest the targets, PLE and MTP fc."""

    from tensorfold.cuda.nvfp4 import thor_dense

    faces = thor_dense.faces(w)
    by_layer: dict = {}
    for name, lin, f32 in faces:
        where, kind = name.rsplit(".", 1) if not name.startswith("mtp.fc") else ("mtp", name.split(".")[-1])
        by_layer.setdefault(kind, []).append((where, name, lin, f32))
    keep = {}
    for kind, items in by_layer.items():
        layered = [(int(where[5:].split(".")[0]), name, lin, f32) for where, name, lin, f32 in items
                   if where.startswith("layer")]
        if kind in ("ple_key", "ple_value", "fc_e", "fc_h") or not layered:
            picks = [(name, lin, f32) for _, name, lin, f32 in items][:1]
        else:
            picks = []
            for t in targets:
                best = min(layered, key=lambda it: abs(it[0] - t))
                if all(best[1] != p[0] for p in picks):
                    picks.append(best[1:])
        for name, lin, f32 in picks:
            if getattr(lin, "thor", False):
                keep[id(lin)] = (name, lin, f32)
    return keep


def exact(lin) -> torch.Tensor:
    """The weight in float64 from TensorFold's own readers (MXFP8: ``w8_rows`` x 2^(e - 127)), not thor_dense's."""

    if getattr(lin, "w8", None) is not None:
        return lin.w8_rows().double() * torch.exp2(lin.scale_rows().double() - 127).repeat_interleave(32, dim=1)
    return lin.weight.double()


def both(x: torch.Tensor, lin) -> tuple[torch.Tensor, torch.Tensor]:
    """thor_dense dequantizing at the call, then on a bf16 copy (a copy the engine holds is put back after)."""

    from tensorfold.cuda.nvfp4 import thor_dense

    held = getattr(lin, "thor_w", None)
    mx = getattr(lin, "w8", None) is not None
    if held is not None:
        del lin.thor_w
    try:
        each_call = thor_dense.linear(x, lin)
        if mx:
            lin.thor_w = thor_dense.dequantize(lin)
        copied = thor_dense.linear(x, lin)
    finally:
        if mx:
            del lin.thor_w
        if held is not None:
            lin.thor_w = held
    return each_call, copied


def portable(x: torch.Tensor, lin, f32: bool) -> torch.Tensor:
    """TensorFold's prompt GEMM for the linear (thor_dense off)."""

    from tensorfold.cuda.nvfp4 import thor_dense
    from tensorfold.families.qwen4_exp.cuda import bf16

    thor_dense.set_active(False)
    try:
        if getattr(lin, "w8", None) is not None:
            return lin.prefill(x)
        return bf16.matmul(x, lin, f32=f32)
    finally:
        thor_dense.set_active(True)


def main() -> None:
    import prefill_profile as pp

    from tensorfold.cuda.nvfp4 import thor_dense

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--layers", default="0,23,47")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--kv-dtype", default="int8")
    args = ap.parse_args()
    t0 = time.perf_counter()
    engine = pp.build_engine(args.model, args.parallel, args.kv_dtype)
    if not thor_dense.S.active:
        raise SystemExit("thor_dense is off in this process (sm_110 and TENSORFOLD_THOR_DENSE on only)")
    targets = chosen(engine.w, [int(v) for v in args.layers.split(",")])
    caught: dict = {}
    original = thor_dense.linear

    def catching(x, lin, out=None):
        if id(lin) in targets and len(caught.setdefault(id(lin), [])) < 2:
            caught[id(lin)].append(x.detach().clone())
        return original(x, lin, out=out)

    thor_dense.linear = catching
    try:
        pp.run_prompt(engine, pp.prompt_ids(args.model, 4096, 777, natural=True))
    finally:
        thor_dense.linear = original
    torch.backends.cuda.matmul.allow_tf32 = False
    cells, worst = [], 0.0
    for key, (name, lin, f32) in targets.items():
        rows = caught.get(key, [])
        if not rows:
            print(f"[ga] {name}: no rows caught", flush=True)
            continue
        x = torch.cat(rows)[:max(M_CELLS)].contiguous()
        wd = exact(lin)
        full = None
        for m in M_CELLS:
            if m > x.shape[0]:
                continue
            xm = x[:m]
            ref = xm.double() @ wd.t()
            norm = float(ref.norm())

            def err(y):
                return float((y.double() - ref).norm()) / norm

            each_call, copied = both(xm, lin)
            if full is None:
                full = thor_dense.linear(x, lin)
            tf = portable(xm, lin, f32)
            e_tf, e_k2 = err(tf), err(each_call)
            ratio = e_k2 / e_tf if e_tf else float("inf")
            worst = max(worst, ratio)
            cell = {"name": name, "m": m, "n": int(lin.n), "k": int(lin.k), "fp32_sums": f32,
                    "err_tensorfold": e_tf, "err_thor": e_k2, "ratio": ratio, "pass": ratio <= 1.5,
                    "copy_equals_dequant": bool(torch.equal(copied, each_call)),
                    "rows_invariant": bool(torch.equal(each_call, full[:m]))}
            cells.append(cell)
        last = [c for c in cells if c["name"] == name]
        print(f"[ga] {name} [{int(lin.n)} x {int(lin.k)}]: ratio " + ", ".join(
            f"M={c['m']} {c['ratio']:.3f}" for c in last) + f"; tf err {last[-1]['err_tensorfold']:.2e}; copy==dequant "
            f"{all(c['copy_equals_dequant'] for c in last)}; rows invariant {all(c['rows_invariant'] for c in last)}",
            flush=True)
    summary = {"cells": len(cells), "pass": sum(c["pass"] for c in cells), "worst_ratio": worst,
               "copy_equals_dequant": all(c["copy_equals_dequant"] for c in cells),
               "rows_invariant": all(c["rows_invariant"] for c in cells), "seconds": time.perf_counter() - t0}
    args.out.write_text(json.dumps({"summary": summary, "cells": cells}, indent=1))
    print(f"[ga] {summary['pass']}/{summary['cells']} cells pass (worst ratio {worst:.3f}); copy == dequant "
          f"{summary['copy_equals_dequant']}; rows invariant {summary['rows_invariant']}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
