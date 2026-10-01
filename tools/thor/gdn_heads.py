"""Which DeltaNet heads carry Thor's chunked-chain error, why, and what it does to a prompt's head logits (THOR-K4).

    python tools/thor/gdn_heads.py MODEL_DIR OUT_DIR [--layers 0,1,2] [--tokens 6000] [--e2e 2048,6000,17000]
        [--exclude 1]

Part 1 (heads): one prompt with TensorFold's chain; for the chosen DeltaNet layers the real chain inputs are caught and
each head's read-out error is measured against fp64 for TensorFold's chain, for Thor's chunked chain (fp16 q, k, v), and
for TensorFold's chain on fp16-rounded q and k (the share of the inputs' rounding), with each head's gate and update
statistics (g, beta), read-out size and the size of q.k against |q||k| (cancellation).
Part 2 (end to end): prompts of the ``--e2e`` lengths through TensorFold's chain, Thor's, Thor's with the ``--exclude``
DeltaNet layers on TensorFold's chain, and the exact-DeltaNet referee (gdn_reference.install_prompt_chain): the head
logits' distance of each from the referee, and the first tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import gdn_reference as ref  # noqa: E402
import ngram_residency as nr  # noqa: E402
import prefill_profile as pp  # noqa: E402


def per_head(y: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    d = (y.double() - r).pow(2).sum(dim=(0, 2)).sqrt()
    return d / r.pow(2).sum(dim=(0, 2)).sqrt().clamp_min(1e-300)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--layers", default="0,1,2")
    ap.add_argument("--tokens", type=int, default=6000)
    ap.add_argument("--e2e", default="2048,6000,17000")
    ap.add_argument("--exclude", default="1")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from tensorfold.cuda.kernels import gdn as tf_gdn
    from tensorfold.cuda.kernels import thor_gdn
    from tensorfold.families.qwen4_exp.cuda import forward, gdn_io
    from tensorfold.families.qwen4_exp.cuda import multi as multi_mod

    engine = pp.build_engine(args.model, 4, "int8")
    if not thor_gdn.S.ready:
        raise SystemExit("thor_gdn is not prepared in this process")
    report: dict = {"heads": {}, "e2e": []}
    layers = [int(x) for x in args.layers.split(",")]
    caught: dict = {li: [] for li in layers}
    orig_chain, orig_front = forward._prefill_chain, gdn_io.front

    def catching(g, st, li, b, a0, a1, c, cut=None):
        if li not in caught:
            return orig_chain(g, st, li, b, a0, a1, c, cut)
        seen: dict = {}

        def front(*a, **kw):
            seen["x"] = orig_front(*a, **kw)
            return seen["x"]

        gdn_io.front = front
        try:
            orig_chain(g, st, li, b, a0, a1, c, cut)
        finally:
            gdn_io.front = orig_front
        caught[li].append(seen["x"])

    thor_gdn.set_active(False)
    forward._prefill_chain = catching
    nr.forget(engine)
    got = pp.run_prompt(engine, pp.prompt_ids(args.model, args.tokens, 2718))
    forward._prefill_chain = orig_chain
    if got["error"]:
        raise RuntimeError(f"the capture prompt failed: {got['error']}")
    print(f"[heads] caught passes: {({li: len(v) for li, v in caught.items()})}", flush=True)
    for li in layers:
        q, k, v, g, beta = (torch.cat([x[i] for x in caught[li]]) for i in range(5))
        s0 = torch.zeros((v.shape[1], 128, 128), dtype=torch.float32, device=v.device)
        y_ref, _ = ref.chain64(q, k, v, g, beta, s0)
        # TensorFold's chain over the whole prompt (it keeps its bits at any split), Thor's in 2048-row passes
        y_tf = tf_gdn.chain(q, k, v, g, beta, s0, torch.empty_like(s0))
        y_tf16 = tf_gdn.chain(q.half().float().contiguous(), k.half().float().contiguous(), v, g, beta, s0,
                              torch.empty_like(s0))
        ys, s = [], s0
        for a in range(0, q.shape[0], 2048):
            fin = torch.empty_like(s0)
            ys.append(thor_gdn.chain(q[a:a + 2048].half(), k[a:a + 2048].half(), v[a:a + 2048].half(),
                                     g[a:a + 2048].contiguous(), beta[a:a + 2048].contiguous(), s, fin))
            s = fin
        y_th = torch.cat(ys)
        e_tf, e_th, e_tf16 = per_head(y_tf, y_ref), per_head(y_th, y_ref), per_head(y_tf16, y_ref)
        e_fp16 = per_head(y_ref.half(), y_ref)                  # fp16 rounding of the exact read-out alone
        hv, hk = v.shape[1], q.shape[1]
        qr, kr = q.repeat_interleave(hv // hk, 1).double(), k.repeat_interleave(hv // hk, 1).double()
        qk = (qr * kr).sum(-1).abs() / (qr.norm(dim=-1) * kr.norm(dim=-1)).clamp_min(1e-300)
        rows = []
        for h in range(hv):
            rows.append({"head": h, "tf": float(e_tf[h]), "thor": float(e_th[h]), "tf_fp16_qk": float(e_tf16[h]),
                         "fp16_floor": float(e_fp16[h]), "x": float(e_th[h] / e_tf[h]),
                         "g_mean": float(g[:, h].mean()), "g_min": float(g[:, h].min()),
                         "g_max": float(g[:, h].max()), "beta_mean": float(beta[:, h].mean()),
                         "y_rms": float(y_ref[:, h].pow(2).mean().sqrt()),
                         "v_rms": float(v[:, h].float().pow(2).mean().sqrt()),
                         "qk_cos_median": float(qk[:, h].median())})
        rows.sort(key=lambda r: -r["x"])
        report["heads"][str(li)] = rows
        print(f"[heads] DeltaNet {li}: layer y tf {float((y_tf.double() - y_ref).norm() / y_ref.norm()):.2e} thor "
              f"{float((y_th.double() - y_ref).norm() / y_ref.norm()):.2e}; worst heads (thor/tf):", flush=True)
        for r in rows[:4]:
            print(f"[heads]   head {r['head']:2d}: tf {r['tf']:.2e} thor {r['thor']:.2e} (x{r['x']:.2f}); tf on fp16 "
                  f"q,k {r['tf_fp16_qk']:.2e}; fp16 floor {r['fp16_floor']:.2e}; g mean {r['g_mean']:.3f} "
                  f"[{r['g_min']:.3f}, {r['g_max']:.3f}], beta {r['beta_mean']:.3f}; y rms {r['y_rms']:.3e}, v rms "
                  f"{r['v_rms']:.3f}; |cos qk| median {r['qk_cos_median']:.3f}", flush=True)
        caught[li] = []
    (args.out / "gdn_heads.json").write_text(json.dumps(report, indent=1))

    # end to end: head logits against the exact-DeltaNet referee
    exclude = {int(x) for x in args.exclude.split(",") if x}
    heads: dict = {}
    joined = multi_mod.MultiDecoder._joined

    def catch(dec, pieces, h, lasts, spent):
        if h is not None:
            heads["last"] = h.float().clone()
        return joined(dec, pieces, h, lasts, spent)

    def excluding(g, st, li, b, a0, a1, c, cut=None):
        if li in exclude and thor_gdn.S.active:
            thor_gdn.S.active = False
            try:
                return orig_chain(g, st, li, b, a0, a1, c, cut)
            finally:
                thor_gdn.S.active = True
        return orig_chain(g, st, li, b, a0, a1, c, cut)

    multi_mod.MultiDecoder._joined = catch
    shared_chain = tf_gdn.chain
    for n in [int(x) for x in args.e2e.split(",") if x]:
        ids = pp.prompt_ids(args.model, n, 31337 + n)
        out: dict = {}
        for name in ("stock", "thor", "thor_excl", "referee"):
            nr.forget(engine)
            thor_gdn.set_active(name in ("thor", "thor_excl"))
            forward._prefill_chain = excluding if name == "thor_excl" else orig_chain
            if name == "referee":
                ref.install_prompt_chain()
            got = pp.run_prompt(engine, ids)
            if got["error"]:
                raise RuntimeError(f"{name} {n}: {got['error']}")
            out[name] = {"first": got["first"], "logits": heads.pop("last")}
            tf_gdn.chain = shared_chain
            forward._prefill_chain = orig_chain
        r_ = out["referee"]["logits"]
        row = {"tokens": len(ids), "first": {k: v["first"] for k, v in out.items()}}
        for name in ("stock", "thor", "thor_excl"):
            d = out[name]["logits"] - r_
            row[name] = {"rms": float(d.pow(2).mean().sqrt()), "max": float(d.abs().max())}
        report["e2e"].append(row)
        print(f"[e2e] {len(ids)} tokens: first {row['first']}; head logits vs the exact-DeltaNet referee: stock rms "
              f"{row['stock']['rms']:.4f}, thor {row['thor']['rms']:.4f}, thor with DeltaNet {sorted(exclude)} on "
              f"TensorFold's chain {row['thor_excl']['rms']:.4f}", flush=True)
        (args.out / "gdn_heads.json").write_text(json.dumps(report, indent=1))
    multi_mod.MultiDecoder._joined = joined
    thor_gdn.set_active(True)


if __name__ == "__main__":
    main()
