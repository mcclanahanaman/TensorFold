"""Are a row's output bits independent of the call it runs in? TensorFold's kernels promise it (drafted == serial,
concurrent == solo); this checks FlashInfer's W4A16 grouped GEMMs on real weights.

    FLASHINFER_NO_DOWNLOAD=1 python tools/thor/row_invariance.py --model /hf/... --out /out/row_invariance.json

8 rows (fixed routing and weights) run alone (M = 8) and inside a 2048-row call, under every tactic of the decode
and prompt sweeps; the combined outputs of those 8 rows are compared bit for bit against the first run. TensorFold's
own kernel is checked the same way (prompt plan, bf16 slots, alone vs inside the big call).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import moe_common as mc  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="/out/row_invariance.json")
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--big", type=int, default=2048)
    args = ap.parse_args()

    import fi_w4a16_moe as fw
    from nvfp4_moe_phase_a import TensorFoldPath

    layer = mc.Checkpoint(args.model).layer(args.layer)
    fi = fw.FlashInferW4A16(layer)
    tf = TensorFoldPath(layer)
    xs = mc.activations(8, seed=77)
    ids_s, w_s = mc.route(xs, layer.router)
    xb = mc.activations(args.big, seed=78)
    ids_b, w_b = mc.route(xb, layer.router)
    xb[:8], ids_b[:8], w_b[:8] = xs, ids_s, w_s
    rep = {"fi": {}, "tf": {}, "errors": {}}

    runs = []
    for label, x, ids, w in (("alone", xs, ids_s, w_s), ("in_big", xb, ids_b, w_b)):
        for tac in fw.sweep_tactics(8) + fw.sweep_tactics(args.big):
            fi.tactic = tac
            try:
                out = fi(x, ids, w)[:8].clone()
                torch.cuda.synchronize()
                runs.append((f"{label}:{tac}", out))
            except Exception as exc:  # noqa: BLE001
                rep["errors"][f"{label}:{tac}"] = repr(exc)[:300]
    base = runs[0][1]
    for name, out in runs:
        rep["fi"][name] = {"bitwise_equal_to_first": bool(torch.equal(out.view(torch.int16), base.view(torch.int16))),
                           "max_abs": float((out.float() - base.float()).abs().max())}
    fi_all = all(v["bitwise_equal_to_first"] for v in rep["fi"].values())

    picks_s, wts_s = tf.prepare(xs, ids_s, w_s)
    picks_b, wts_b = tf.prepare(xb, ids_b, w_b)
    a = tf(xs, picks_s, wts_s)[:8].clone()
    b = tf(xb, picks_b, wts_b)[:8].clone()
    torch.cuda.synchronize()
    rep["tf"] = {"bitwise_equal": bool(torch.equal(a.view(torch.int16), b.view(torch.int16))),
                 "max_abs": float((a.float() - b.float()).abs().max())}
    rep["summary"] = {"fi_runs": len(runs), "fi_all_bitwise_equal": fi_all,
                      "fi_max_abs_over_runs": max(v["max_abs"] for v in rep["fi"].values()),
                      "tf_alone_vs_in_big_bitwise_equal": rep["tf"]["bitwise_equal"]}
    Path(args.out).write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep["summary"]), flush=True)
    for name, v in rep["fi"].items():
        if not v["bitwise_equal_to_first"]:
            print("  differs:", name, v, flush=True)
    print("errors:", list(rep["errors"])[:5], flush=True)


if __name__ == "__main__":
    main()
