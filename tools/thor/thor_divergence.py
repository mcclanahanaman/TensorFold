"""G-B for the Thor experts: where two greedy runs diverge, how close the call was, and which run a higher-precision
referee sides with.

    python tools/thor/thor_divergence.py collect BASE_URL OUT.json [--top 5] [--only NAME,...]
    python tools/thor/thor_divergence.py compare STOCK.json THOR.json [--out DIFF.json]
    python tools/thor/thor_divergence.py reference MODEL STOCK.json THOR.json OUT.json [--context 131072]
        [--dense] [--kv-dtype int8] [--save-logits LOGITS.pt]
    python tools/thor/thor_divergence.py logits MODEL STOCK.json THOR.json OUT.pt [--kv-dtype int8]
    python tools/thor/thor_divergence.py distance REFEREE.json REFEREE.pt A.pt B.pt [--out DIST.json]
    python tools/thor/thor_divergence.py selfcheck MODEL [--layer 23]

``collect`` is a client (a running TensorFold server): the probe prompts of tools/thor_equivalence_probe.py (6 long
ones of 22k-60k characters asking for 300 tokens, 4 short), greedy, thinking off, with each token's id, log
probability and top-``--top`` alternatives. ``compare`` finds each prompt's first differing token and reports, at
that position, both runs' tokens and their top-2 log-probability margins (a near-tie is a margin of a few hundredths).
``reference`` loads the model in-process on the Thor layout with the fp32 referee (tools/thor/moe_reference.py:
fp32 SwiGLU, fp32 expert outputs, everything else TensorFold's own prompt arithmetic), prefills each divergent
prompt plus the common prefix, and reports the referee's log probabilities of both runs' tokens there: the run whose
token the referee ranks higher "agrees". ``--dense`` (THOR-K2) also makes every dense prompt GEMM exact
(tools/thor/dense_reference.py: float64 on the exact weights, rounded once to each buffer's dtype), so neither run's
GEMM arithmetic is the referee's; ``--save-logits`` keeps the referee's full next-token logits at each divergence (the
head's float64 logits with ``--dense``). ``logits`` saves a run's own full next-token logits on the same prefixes
(this process's environment picks the run: TENSORFOLD_THOR_FI_MOE / TENSORFOLD_THOR_DENSE), and ``distance`` gives,
per divergence, the referee's log probabilities of both tokens and each run's distance from the referee's logits.
``selfcheck`` checks the routed referee against an fp64 computation on real weights. Run ``reference``, ``logits``
and ``selfcheck`` on Thor through the runner (sm_110; ``reference`` needs FlashInfer importable).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# -- the probe prompts (the same text as tools/thor_equivalence_probe.py on thor/w1-p1-dense-fp8) ------------------
LONG = (("minutes", "summary", 22000), ("code", "code", 30000), ("table", "extraction", 38000),
        ("minutes", "summary", 46000), ("code", "code", 54000), ("table", "extraction", 60000))
_ASK = {
    "summary": "Summarize the document above in about 300 tokens. Include the decisions, the owners, and the dates.",
    "extraction": ("Extract every row whose amount is above 1000. "
                   "Reply in about 300 tokens as a list of id, name, and amount."),
    "code": "Find the bug in the functions above and write a corrected version. Use about 300 tokens.",
}


def _fit(body: str, n: int) -> str:
    if len(body) >= n:
        return body[:n]
    return (body * ((n + len(body) - 1) // len(body)))[:n]


def _minutes(n: int) -> str:
    rows, i = [], 0
    while sum(len(row) for row in rows) < n:
        i += 1
        day = (i % 28) + 1
        rows.append(f"Item {i}. On 2026-09-{day:02d} the committee decided to approve budget line {1000 + i} "
                    f"owned by person{i % 50}. The vote was {i % 7} to {(i + 3) % 5}. "
                    f"Risk: a delay of {i % 12} weeks if vendor {i % 9} misses milestone M{i % 6}.\n")
    return _fit("".join(rows), n)


def _code(n: int) -> str:
    rows, i = [], 0
    while sum(len(row) for row in rows) < n:
        i += 1
        rows.append(f"def step_{i}(value, limit):\n"
                    f"    # module {i}: scale by {1 + (i % 5)} then fold the running total\n"
                    f"    total = value\n"
                    f"    for k in range({2 + (i % 4)}):\n"
                    f"        total = total * {1 + (i % 5)} + k - limit\n"
                    f"    if total > limit:\n"
                    f"        return total / 0\n"
                    f"    return total\n\n")
    return _fit("".join(rows), n)


def _table(n: int) -> str:
    rows, i = ["id,name,dept,amount,status\n"], 0
    while sum(len(row) for row in rows) < n:
        i += 1
        rows.append(f"{i},name{i},dept{i % 8},{100 + (i * 37) % 4000},{'open' if i % 3 else 'closed'}\n")
    return _fit("".join(rows), n)


def prompts() -> list[dict]:
    bodies = {"minutes": _minutes, "code": _code, "table": _table}
    items = [{"name": f"long-{i}-{kind}", "max_tokens": 300, "prompt": bodies[kind](chars) + "\n\n" + _ASK[task]}
             for i, (kind, task, chars) in enumerate(LONG)]
    for name, text in (("short-sum", "What is 17 plus 4? Reply with the number only."),
                       ("short-color", "Name the usual color of a clear daytime sky, in one word."),
                       ("short-haiku", "Write a haiku about rain on a tin roof."),
                       ("short-primes", "List three prime numbers greater than 10.")):
        items.append({"name": name, "max_tokens": 32, "prompt": text})
    return items


def body(prompt: str, max_tokens: int, top: int) -> dict:
    return {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0,
            "logprobs": True, "top_logprobs": top, "return_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False}}


# -- collect ---------------------------------------------------------------------------------------------------------
def collect(base: str, out: Path, top: int, only: set[str]) -> None:
    base = base.rstrip("/")
    with urllib.request.urlopen(base + "/v1/models", timeout=60) as resp:
        model = json.loads(resp.read())["data"][0]["id"]
    rows = []
    for item in prompts():
        if only and item["name"] not in only:
            continue
        payload = {"model": model, **body(item["prompt"], item["max_tokens"], top)}
        req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=3600) as resp:
            result = json.loads(resp.read())
        choice = result["choices"][0]
        content = (choice.get("logprobs") or {}).get("content") or []
        ids = result.get("tensorfold", {}).get("token_ids") or []
        rows.append({"name": item["name"], "max_tokens": item["max_tokens"], "elapsed_s": time.time() - t0,
                     "text": choice["message"].get("content"), "token_ids": ids,
                     "logprobs": [{"token": c.get("token"), "logprob": c.get("logprob"),
                                   "top": [(t.get("token"), t.get("logprob")) for t in c.get("top_logprobs") or []]}
                                  for c in content]})
        print(f"{item['name']}: {len(ids)} tokens in {time.time() - t0:.1f}s", flush=True)
        out.write_text(json.dumps({"base": base, "model": model, "rows": rows}, indent=1))


# -- compare ---------------------------------------------------------------------------------------------------------
def margin(entry: dict) -> float | None:
    top = [lp for _, lp in entry.get("top") or []]
    return (top[0] - top[1]) if len(top) >= 2 else None


def compare(a_path: Path, b_path: Path) -> list[dict]:
    a_rows = {r["name"]: r for r in json.loads(a_path.read_text())["rows"]}
    b_rows = {r["name"]: r for r in json.loads(b_path.read_text())["rows"]}
    out = []
    for name, a in a_rows.items():
        b = b_rows.get(name)
        if b is None:
            continue
        ta, tb = a["token_ids"], b["token_ids"]
        i = next((k for k, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
        if i is None and len(ta) != len(tb):
            i = min(len(ta), len(tb))
        row = {"name": name, "equal": i is None, "tokens": [len(ta), len(tb)]}
        if i is not None:
            la = a["logprobs"][i] if i < len(a["logprobs"]) else {}
            lb = b["logprobs"][i] if i < len(b["logprobs"]) else {}
            alt_b = dict(lb.get("top") or [])
            alt_a = dict(la.get("top") or [])
            row.update({"index": i, "a": {"id": ta[i] if i < len(ta) else None, "token": la.get("token"),
                                          "logprob": la.get("logprob"), "top2_margin": margin(la),
                                          "logprob_of_b_token": alt_a.get(lb.get("token"))},
                        "b": {"id": tb[i] if i < len(tb) else None, "token": lb.get("token"),
                              "logprob": lb.get("logprob"), "top2_margin": margin(lb),
                              "logprob_of_a_token": alt_b.get(la.get("token"))}})
        out.append(row)
    return out


def print_compare(rows: list[dict]) -> None:
    for r in rows:
        if r["equal"]:
            print(f"{r['name']:14s} equal ({r['tokens'][0]} tokens)")
            continue
        a, b = r["a"], r["b"]
        print(f"{r['name']:14s} first divergence at token {r['index']}: A {a['token']!r} (lp {a['logprob']}, "
              f"top-2 margin {a['top2_margin']}) vs B {b['token']!r} (lp {b['logprob']}, margin {b['top2_margin']}); "
              f"A's lp of B's token {a['logprob_of_b_token']}, B's lp of A's token {b['logprob_of_a_token']}")


# -- reference -------------------------------------------------------------------------------------------------------
def _engine(model: str, context: int, kv_dtype: str = "bf16", thor_layout: bool = True):
    import torch  # noqa: F401

    from tensorfold.cuda.nvfp4 import thor_experts as thor
    from tensorfold.cuda.server import App
    from tensorfold.families.qwen4_exp import cuda_engine

    if thor_layout and not thor.available():
        raise SystemExit("the referee reads the Thor layout: run on Thor (sm_110) with FlashInfer importable")
    engine = cuda_engine(model, context=context, context_explicit=True, no_drafts=True, kv_dtype=kv_dtype)
    return engine, App(engine, Path(model), "thor-divergence", context_window=context)


def _capture() -> dict:
    """The last prompt pass's head logits (as the head stored them), in ``box["logits"]`` (float32 [V])."""

    from tensorfold.families.qwen4_exp.cuda import forward

    box: dict = {}
    original = forward.finish

    def finish(w, mixer, b, R, pending, logits=True, ends=()):
        out = original(w, mixer, b, R, pending, logits=logits, ends=ends)
        if b.prefill and out is not None:
            box["logits"] = out[0].float().detach().clone()
        return out

    forward.finish = finish
    return box


def _prefixes(a_path: Path, b_path: Path, app) -> list[tuple[dict, list[int]]]:
    """(compare row, prompt + the common prefix) for each prompt where the runs diverge."""

    a_rows = {r["name"]: r for r in json.loads(a_path.read_text())["rows"]}
    items = {p["name"]: p for p in prompts()}
    out = []
    for r in compare(a_path, b_path):
        if r["equal"]:
            continue
        item = items[r["name"]]
        prompt = app.prepare(body(item["prompt"], item["max_tokens"], 5), True).prompt
        out.append((r, list(prompt) + list(a_rows[r["name"]]["token_ids"][:r["index"]])))
    return out


def _faithful(engine, ids: list[int], generated: int) -> tuple:
    """The run's own next-token logits as its server computed them: the prompt (``ids`` less its last ``generated``
    tokens) in prompt passes, then each generated token through a one-row decode step (the bits of MTP windows and
    concurrent rounds: TensorFold's drafted == serial and solo == concurrent). Returns (first token, logits)."""

    from tensorfold.families.qwen4_exp.cuda.decode import prefill
    from tensorfold.families.qwen4_exp.cuda.forward import commit

    e = engine.e
    prompt, tail = ids[:len(ids) - generated], ids[len(ids) - generated:]
    first = prefill(e, prompt, None, mtp=False)
    out = None
    for tok in tail:
        out = e.forward([tok])[0].float().clone()
        commit(e.w, e.st, e.buf, 1, 1)
    return int(first), out


def logits(model: str, a_path: Path, b_path: Path, out_path: Path, context: int, kv_dtype: str,
           faithful: bool = False) -> None:
    """This process's run (its environment's switches) on each divergence's prefix: full next-token logits, the
    prefix in one prompt pass, or (``faithful``) the prompt in prompt passes and the generated tokens decoded."""

    import os

    import torch

    engine, app = _engine(model, context, kv_dtype, thor_layout=False)
    box = _capture()
    saved = {"env": {k: os.environ.get(k) for k in ("TENSORFOLD_THOR_FI_MOE", "TENSORFOLD_THOR_DENSE",
                                                     "TENSORFOLD_THOR_DENSE_WEIGHTS")}, "kv_dtype": kv_dtype,
             "faithful": faithful, "rows": {}}
    for r, ids in _prefixes(a_path, b_path, app):
        t0 = time.time()
        row = {"prefix_tokens": len(ids), "index": r["index"]}
        if faithful:
            first, out = _faithful(engine, ids, r["index"])
            row.update({"logits": out.cpu(), "first": first, "argmax": int(out.argmax())})
        else:
            top = _next_token_top(engine, ids)
            row.update({"logits": box.pop("logits").cpu(), "argmax": int(top["id"])})
        row["argmax_is"] = "A" if row["argmax"] == r["a"]["id"] else "B" if row["argmax"] == r["b"]["id"] else "other"
        saved["rows"][r["name"]] = row
        print(f"{r['name']:14s} logits saved: argmax {row['argmax']} ({row['argmax_is']}'s token"
              + (f"; first token {row['first']}" if faithful else "")
              + f"; {time.time() - t0:.0f}s, {len(ids)} tokens)", flush=True)
    torch.save(saved, str(out_path))


def _next_token_top(engine, ids: list[int], top: int = 20) -> dict:
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.engine.probabilities import Probabilities

    probabilities = Probabilities(top, len(ids), 1)
    out: list[int] = []
    engine.generate(ids, 1, Sampling(seed=0, temperature=0.0), lambda new: out.extend(new) or False,
                    probabilities=probabilities)
    return probabilities.emitted(out[:1])[0]


def reference(model: str, a_path: Path, b_path: Path, out_path: Path, context: int, dense: bool = False,
              kv_dtype: str = "bf16", save_logits: Path | None = None) -> None:
    import torch

    import moe_reference

    diff = compare(a_path, b_path)
    engine, app = _engine(model, context, kv_dtype)
    moe_reference.install()
    if dense:
        import dense_reference

        dense_reference.install(engine.w)
    box = _capture()
    prefixes = {r["name"]: ids for r, ids in _prefixes(a_path, b_path, app)}
    saved: dict = {"dense": dense, "kv_dtype": kv_dtype, "rows": {}}
    results = []
    for r in diff:
        if r["equal"]:
            results.append(r)
            continue
        ids = prefixes[r["name"]]
        t0 = time.time()
        ref = _next_token_top(engine, ids)
        stored = box.pop("logits", None)
        if save_logits is not None and stored is not None:
            head = dense_reference.LAST.pop("head")[0].cpu() if dense else None
            saved["rows"][r["name"]] = {"logits": stored.cpu(), "logits_f64": head, "argmax": int(ref["id"]),
                                        "prefix_tokens": len(ids), "index": r["index"]}
            torch.save(saved, str(save_logits))
        top = {int(i): float(v) for i, v in ref["top"]}
        a_id, b_id = r["a"]["id"], r["b"]["id"]
        la, lb = top.get(a_id), top.get(b_id)
        agrees = None if la is None and lb is None else ("A" if (lb is None or (la is not None and la > lb)) else "B")
        r = {**r, "reference": {"top": ref["top"][:5], "logprob_a_token": la, "logprob_b_token": lb,
                                "argmax": int(ref["id"]), "agrees_with": agrees, "seconds": time.time() - t0,
                                "prefix_tokens": len(ids), "dense": dense, "kv_dtype": kv_dtype}}
        print(f"{r['name']:14s} referee: A's token lp {la}, B's token lp {lb}, argmax {ref['id']} -> agrees with "
              f"{agrees} ({time.time() - t0:.0f}s, {len(ids)} tokens)", flush=True)
        results.append(r)
        out_path.write_text(json.dumps({"a": str(a_path), "b": str(b_path), "rows": results}, indent=1))
    out_path.write_text(json.dumps({"a": str(a_path), "b": str(b_path), "rows": results}, indent=1))


def distance(ref_json: Path, ref_pt: Path, a_pt: Path, b_pt: Path) -> list[dict]:
    """Per divergence: the referee's log probabilities of A's and B's tokens (from its full logits: float64 when it
    kept them), which it agrees with, and each run's distance from its logits (max and RMS of the logit difference,
    KL(referee || run), each one's logit margin of A's token over B's)."""

    import torch

    rows = {r["name"]: r for r in json.loads(ref_json.read_text())["rows"] if not r.get("equal")}
    ref, a, b = (torch.load(str(p), weights_only=False)["rows"] for p in (ref_pt, a_pt, b_pt))
    out = []
    for name, r in rows.items():
        if name not in ref or name not in a or name not in b:
            continue
        rr = ref[name]
        z = (rr["logits_f64"] if rr.get("logits_f64") is not None else rr["logits"]).double()
        lp = torch.log_softmax(z, dim=-1)
        ia, ib = int(r["a"]["id"]), int(r["b"]["id"])
        la, lb = float(lp[ia]), float(lp[ib])
        row = {"name": name, "index": r["index"], "a_token": r["a"]["token"], "b_token": r["b"]["token"],
               "referee": {"logprob_a_token": la, "logprob_b_token": lb, "margin_a_over_b": float(z[ia] - z[ib]),
                           "argmax": int(z.argmax()), "precision": "float64" if rr.get("logits_f64") is not None
                           else "bf16", "agrees_with": "A" if la > lb else "B" if lb > la else "tie"}}
        for label, run in (("A", a[name]), ("B", b[name])):
            x = run["logits"].double()
            d = x - z
            row[label] = {"max_abs": float(d.abs().max()), "rms": float(d.pow(2).mean().sqrt()),
                          "kl_ref_run": float((lp.exp() * (lp - torch.log_softmax(x, dim=-1))).sum()),
                          "margin_a_over_b": float(x[ia] - x[ib]), "argmax": int(x.argmax()),
                          "argmax_is": run.get("argmax_is")}
        out.append(row)
    return out


def print_distance(rows: list[dict]) -> None:
    for r in rows:
        ref, a, b = r["referee"], r["A"], r["B"]
        print(f"{r['name']:14s} token {r['index']}: referee ({ref['precision']}) lp A {r['a_token']!r} "
              f"{ref['logprob_a_token']:.5f}, B {r['b_token']!r} {ref['logprob_b_token']:.5f} (margin A-B "
              f"{ref['margin_a_over_b']:+.5f}) -> {ref['agrees_with']}; distance A max {a['max_abs']:.4f} rms "
              f"{a['rms']:.5f} KL {a['kl_ref_run']:.2e} (margin {a['margin_a_over_b']:+.4f}), B max "
              f"{b['max_abs']:.4f} rms {b['rms']:.5f} KL {b['kl_ref_run']:.2e} (margin {b['margin_a_over_b']:+.4f})",
              flush=True)


def selfcheck(model: str, layer_index: int) -> None:
    """The referee's routed outputs against fp64 on one real layer: relative error, expected ~1e-6."""

    import torch

    import moe_common as mc
    import moe_reference
    from tensorfold.cuda import experts as grouped
    from tensorfold.cuda.nvfp4 import thor_experts as thor

    layer = mc.Checkpoint(model).layer(layer_index)
    p = lambda q: (q.w, q.s, q.s2)  # noqa: E731
    ex = thor.make(p(layer.gate), p(layer.up), p(layer.down))
    ref_w = mc.reference_weights(layer)
    for m in (5, 300):
        x = mc.activations(m, seed=m)
        ids, w = mc.route(x, layer.router)
        picks = torch.cat([ids, torch.full((m, 1), mc.E, dtype=torch.int32, device="cuda")], 1).contiguous()
        plan = grouped.Plan(m, mc.TOP + 1, mc.E + 1, "cuda", prefill=True)
        grouped.route(picks, plan, moe_reference.TILE)
        y32 = torch.zeros(m, mc.TOP + 1, mc.D, device="cuda")
        h32 = torch.zeros(m * (mc.TOP + 1), mc.NI, device="cuda")
        moe_reference.routed(x, ex, plan, y32, h32, m, mc.TOP)
        worst = 0.0
        for r in range(0, m, max(1, m // 5)):
            for s in range(mc.TOP):
                e = int(ids[r, s])
                xv = x[r].double()
                g, u = xv @ ref_w["gate"][e].double().t(), xv @ ref_w["up"][e].double().t()
                want = (g / (1 + torch.exp(-g)) * u) @ ref_w["down"][e].double().t()
                worst = max(worst, float((y32[r, s].double() - want).norm() / want.norm()))
        print(f"selfcheck M={m}: worst per-pair relative error vs fp64 {worst:.2e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("base")
    c.add_argument("out", type=Path)
    c.add_argument("--top", type=int, default=5)
    c.add_argument("--only", default="")
    d = sub.add_parser("compare")
    d.add_argument("a", type=Path)
    d.add_argument("b", type=Path)
    d.add_argument("--out", type=Path)
    r = sub.add_parser("reference")
    r.add_argument("model")
    r.add_argument("a", type=Path)
    r.add_argument("b", type=Path)
    r.add_argument("out", type=Path)
    r.add_argument("--context", type=int, default=131072)
    r.add_argument("--dense", action="store_true", help="every dense prompt GEMM exact too (dense_reference.py)")
    r.add_argument("--kv-dtype", default="bf16")
    r.add_argument("--save-logits", type=Path, default=None)
    lg = sub.add_parser("logits")
    lg.add_argument("model")
    lg.add_argument("a", type=Path)
    lg.add_argument("b", type=Path)
    lg.add_argument("out", type=Path)
    lg.add_argument("--context", type=int, default=131072)
    lg.add_argument("--kv-dtype", default="bf16")
    lg.add_argument("--faithful", action="store_true", help="the generated tokens through decode steps, as served")
    dd = sub.add_parser("distance")
    dd.add_argument("ref_json", type=Path)
    dd.add_argument("ref_pt", type=Path)
    dd.add_argument("a_pt", type=Path)
    dd.add_argument("b_pt", type=Path)
    dd.add_argument("--out", type=Path)
    s = sub.add_parser("selfcheck")
    s.add_argument("model")
    s.add_argument("--layer", type=int, default=23)
    args = ap.parse_args()
    if args.cmd == "collect":
        collect(args.base, args.out, args.top, {n for n in args.only.split(",") if n})
    elif args.cmd == "compare":
        rows = compare(args.a, args.b)
        print_compare(rows)
        if args.out:
            args.out.write_text(json.dumps(rows, indent=1))
    elif args.cmd == "reference":
        reference(args.model, args.a, args.b, args.out, args.context, args.dense, args.kv_dtype, args.save_logits)
    elif args.cmd == "logits":
        logits(args.model, args.a, args.b, args.out, args.context, args.kv_dtype, args.faithful)
    elif args.cmd == "distance":
        rows = distance(args.ref_json, args.ref_pt, args.a_pt, args.b_pt)
        print_distance(rows)
        if args.out:
            args.out.write_text(json.dumps(rows, indent=1))
    else:
        selfcheck(args.model, args.layer)


if __name__ == "__main__":
    main()
