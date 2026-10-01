"""Flash Next's DeltaNet prompt chain on Jetson Thor: TensorFold's serial fp32 kernel vs FlashInfer's chunked kernels.

    python tools/thor/gdn_prefill_bench.py INPUTS.pt OUT.json

INPUTS.pt is ``prefill_profile.py --save-gdn``'s capture: the first DeltaNet layer's q, k (fp32, normed, q scaled),
v (bf16), g, beta (fp32) and initial state (fp32 [heads, dv, dk]) for two passes of a real prompt. Runs, each on
2048 rows (one pass) and on 4096 (both passes as one chain):

  tensorfold       ``tensorfold.cuda.kernels.gdn.chain`` (what prompts run today)
  tensorfold_bf16  the same kernel on bf16 q and k (the share of the error that bf16 inputs alone cost)
  fi_sm100         FlashInfer's chunked CuTe DSL kernel (``chunk_gated_delta_rule_sm100``) called directly: its
                   public entry dispatches on compute capability 9, 10 and 12, not 11
  fi_cake          FlashInfer's Cake GDN backend through the public entry (``backend="cake_gdn"``)

Each output and final state is compared with an fp64 recurrence of the same rule (decay, delta update, read-out); the
error ratio to ``tensorfold`` is the G-A figure. Times are CUDA events, median of 5 rounds, each path
timed after a second of GEMMs (the clocks drop while CuTe DSL compiles).
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import torch


def reference(q, k, v, g, beta, s0):
    """fp64 serial gated delta rule: s *= g; delta = (v - s k) beta; s += delta k^T; o = s q (per value head)."""

    t, hk, _ = q.shape
    hv = v.shape[1]
    rep = hv // hk
    qd, kd = q.double().repeat_interleave(rep, dim=1), k.double().repeat_interleave(rep, dim=1)
    vd, gd, bd = v.double(), g.double(), beta.double()
    s = s0.double().clone()
    out = torch.empty((t, hv, v.shape[2]), dtype=torch.float64, device=q.device)
    for i in range(t):
        s.mul_(gd[i][:, None, None])
        mem = torch.einsum("hvk,hk->hv", s, kd[i])
        delta = (vd[i] - mem) * bd[i][:, None]
        s.add_(delta[:, :, None] * kd[i][:, None, :])
        out[i] = torch.einsum("hvk,hk->hv", s, qd[i])
    return out, s


def rel(a: torch.Tensor, ref: torch.Tensor) -> float:
    return float((a.double() - ref).norm() / ref.norm())


def spin(seconds: float = 1.0) -> None:
    """GEMMs for ``seconds``: Jetson clocks drop while the host compiles a kernel and ramp back under load."""

    import time

    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    end = time.time() + seconds
    while time.time() < end:
        x @ x
    torch.cuda.synchronize()


def timed(fn, rounds: int = 5, iters: int = 5) -> float:
    fn()
    torch.cuda.synchronize()
    spin()
    out = []
    for _ in range(rounds):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / iters)
    return statistics.median(out)


def main() -> None:
    from tensorfold.cuda.kernels import gdn as tf_gdn

    caps = torch.load(sys.argv[1], map_location="cuda")
    out_path = Path(sys.argv[2])
    a, b = caps[0], caps[1]
    cases = {"2048": a, "4096": {key: (torch.cat([a[key], b[key]]) if key != "state" else a[key]) for key in a}}
    results: dict = {"shapes": {k: list(v.shape) for k, v in a.items()}}
    spin(3.0)                                             # Jetson clocks ramp: a few seconds of work first
    for name, c in cases.items():
        q, k, v, g, beta, s0 = (c[key].contiguous() for key in ("q", "k", "v", "g", "beta", "state"))
        rows, hv = v.shape[0], v.shape[1]
        ref_out, ref_state = reference(q, k, v, g, beta, s0)
        row: dict = {}

        def tf_run(qq=q, kk=k):
            final = torch.empty_like(s0)
            y = tf_gdn.chain(qq, kk, v, g, beta, s0, final)
            return y, final

        for label, fn in (("tensorfold", tf_run),
                          ("tensorfold_bf16", lambda: tf_run(q.bfloat16().contiguous(), k.bfloat16().contiguous()))):
            try:
                y, final = fn()
                torch.cuda.synchronize()
                row[label] = {"ms": timed(fn), "out_err": rel(y, ref_out), "state_err": rel(final, ref_state)}
            except Exception as exc:                          # noqa: BLE001
                row[label] = {"error": repr(exc)[:400]}
        cu = torch.tensor([0, rows], dtype=torch.int32, device="cuda")
        qb, kb = q.bfloat16().contiguous(), k.bfloat16().contiguous()
        try:
            from flashinfer.gdn_kernels import chunk_gated_delta_rule_sm100 as fi100

            if fi100 is None:
                raise RuntimeError("chunk_gated_delta_rule_sm100 is None in this FlashInfer")
            for gate_name, gate in (("linear", g), ("log", torch.log(g.clamp(min=1e-30)))):
                def fi_run(gate=gate):
                    o = torch.empty((rows, hv, v.shape[2]), dtype=torch.bfloat16, device="cuda")
                    st = torch.empty((1, *s0.shape), dtype=torch.float32, device="cuda")
                    fi100(qb, kb, v, gate.contiguous(), beta, o, cu, s0[None].contiguous(), st, 1.0)
                    return o, st[0]
                try:
                    y, final = fi_run()
                    torch.cuda.synchronize()
                    row[f"fi_sm100_{gate_name}_gate"] = {"ms": timed(fi_run), "out_err": rel(y, ref_out),
                                                         "state_err": rel(final, ref_state)}
                except Exception as exc:                      # noqa: BLE001
                    row[f"fi_sm100_{gate_name}_gate"] = {"error": repr(exc)[:600]}
        except Exception as exc:                              # noqa: BLE001
            row["fi_sm100"] = {"error": repr(exc)[:600]}
        try:
            from flashinfer.gdn_prefill import chunk_gated_delta_rule

            for gate_name, gate in (("linear", g), ("log", torch.log(g.clamp(min=1e-30)))):
                def cake_run(gate=gate):
                    return chunk_gated_delta_rule(qb, kb, v, g=gate.contiguous(), beta=beta, scale=1.0,
                                                  initial_state=s0[None].contiguous(), output_final_state=True,
                                                  cu_seqlens=cu, backend="cake_gdn")
                try:
                    y, final = cake_run()
                    torch.cuda.synchronize()
                    row[f"fi_cake_{gate_name}_gate"] = {"ms": timed(cake_run), "out_err": rel(y, ref_out),
                                                        "state_err": rel(final[0], ref_state)}
                except Exception as exc:                      # noqa: BLE001
                    row[f"fi_cake_{gate_name}_gate"] = {"error": repr(exc)[:600]}
        except Exception as exc:                              # noqa: BLE001
            row["fi_cake"] = {"error": repr(exc)[:600]}
        base = row.get("tensorfold", {})
        for r in row.values():
            if "ms" in r and base.get("ms"):
                r["speedup"] = base["ms"] / r["ms"]
                r["out_err_ratio"] = r["out_err"] / base["out_err"] if base.get("out_err") else None
                r["state_err_ratio"] = r["state_err"] / base["state_err"] if base.get("state_err") else None
        results[name] = row
        for label, r in row.items():
            if "ms" in r:
                print(f"[gdn] rows {name} {label}: {r['ms']:.3f} ms (x{r.get('speedup', 1):.2f}), out err "
                      f"{r['out_err']:.2e} (x{r.get('out_err_ratio') or 1:.2f}), state err {r['state_err']:.2e} "
                      f"(x{r.get('state_err_ratio') or 1:.2f})", flush=True)
            else:
                print(f"[gdn] rows {name} {label}: {r.get('error')}", flush=True)
    out_path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
