"""Print the Phase A tables from the harness JSON files: G-A (rel error vs fp32, ratio to TensorFold) and speed.

    python tools/thor/summarize_phase_a.py out/phase_a.json [more.json ...]
"""

from __future__ import annotations

import json
import sys

SKIP = {"load", "h_amax", "x_amax", "ref_fp32_vs_fp64", "fiw4a16_mode", "fiw4a16_tactic", "fiw4a16_launch_error"}


def main() -> None:
    for path in sys.argv[1:]:
        rep = json.load(open(path))
        print(f"== {path}: {rep.get('device')} cc {rep.get('cc')} torch {rep.get('torch')} "
              f"flashinfer {rep.get('flashinfer')}")
        for lk, lv in rep.get("accuracy", {}).items():
            info = lv.get("info", {})
            print(f"-- {lk}: gate/up scale2 equal {info.get('gate_up_scale2_equal')}, "
                  f"tf packing exact {all((info.get('tf_packing_exact') or {'x': None}).values())}")
            ms = [k for k in lv if k.startswith("M")]
            paths = []
            for k in ms:
                for p in lv[k]:
                    if p not in SKIP and isinstance(lv[k][p], dict) and "rel_fro" in lv[k][p] and p not in paths:
                        paths.append(p)
            print(f"   {'path':9s} " + " ".join(f"{m:>22s}" for m in ms))
            for p in paths:
                cells = []
                for k in ms:
                    a, t = lv[k].get(p), lv[k].get("tf")
                    if a is None:
                        cells.append(f"{'-':>22s}")
                        continue
                    r = a["rel_fro"] / t["rel_fro"] if t else float("nan")
                    tag = "" if p == "tf" else (" PASS" if r <= 1.5 else " FAIL")
                    cells.append(f"{a['rel_fro']:.3e} x{r:5.2f}{tag:5s}")
                print(f"   {p:9s} " + " ".join(f"{c:>22s}" for c in cells))
            for k in ms:
                if "ref_fp32_vs_fp64" in lv[k]:
                    print(f"   fp32 reference vs fp64 at {k}: rel {lv[k]['ref_fp32_vs_fp64']['rel_fro']:.2e}")
        sp = rep.get("speed", {})
        if sp:
            paths = []
            for row in sp.values():
                for p, v in row.items():
                    if isinstance(v, dict) and "median_ms" in v and p not in paths:
                        paths.append(p)
            print(f"-- speed (median ms / effective TFLOPS), layer {rep.get('args', {}).get('speed_layer')}")
            print(f"   {'M':>6s} " + " ".join(f"{p:>20s}" for p in paths))
            for mk, row in sp.items():
                cells = []
                for p in paths:
                    v = row.get(p)
                    cells.append(f"{v['median_ms']:9.3f}/{v['tflops']:7.2f}" if v else f"{'-':>17s}")
                print(f"   {mk:>6s} " + " ".join(f"{c:>20s}" for c in cells))
        if rep.get("errors"):
            print("-- errors:")
            for k, v in rep["errors"].items():
                print(f"   {k}: {v.splitlines()[0][:300]}")


if __name__ == "__main__":
    main()
