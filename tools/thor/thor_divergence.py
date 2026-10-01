"""G-B for the Thor experts: where two greedy runs diverge, how close the call was, and which run a higher-precision
referee sides with.

    python tools/thor/thor_divergence.py collect BASE_URL OUT.json [--top 5] [--only NAME,...]
    python tools/thor/thor_divergence.py compare STOCK.json THOR.json [--out DIFF.json]
    python tools/thor/thor_divergence.py reference MODEL STOCK.json THOR.json OUT.json [--context 131072]
    python tools/thor/thor_divergence.py selfcheck MODEL [--layer 23]

``collect`` is a client (a running TensorFold server): the probe prompts of tools/thor_equivalence_probe.py (6 long
ones of 22k-60k characters asking for 300 tokens, 4 short), greedy, thinking off, with each token's id, log
probability and top-``--top`` alternatives. ``compare`` finds each prompt's first differing token and reports, at
that position, both runs' tokens and their top-2 log-probability margins (a near-tie is a margin of a few hundredths).
``reference`` loads the model in-process on the Thor layout with the fp32 referee (tools/thor/moe_reference.py:
fp32 SwiGLU, fp32 expert outputs, everything else TensorFold's own prompt arithmetic), prefills each divergent
prompt plus the common prefix, and reports the referee's log probabilities of both runs' tokens there: the run whose
token the referee ranks higher "agrees". ``selfcheck`` checks the referee against an fp64 computation on real weights.
Run ``reference`` and ``selfcheck`` on Thor through the runner (sm_110, FlashInfer importable).
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
def _engine(model: str, context: int):
    import torch  # noqa: F401

    from tensorfold.cuda.nvfp4 import thor_experts as thor
    from tensorfold.cuda.server import App
    from tensorfold.families.qwen4_exp import cuda_engine

    if not thor.available():
        raise SystemExit("the referee reads the Thor layout: run on Thor (sm_110) with FlashInfer importable")
    engine = cuda_engine(model, context=context, context_explicit=True, no_drafts=True)
    return engine, App(engine, Path(model), "thor-divergence", context_window=context)


def _next_token_top(engine, ids: list[int], top: int = 20) -> dict:
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.engine.probabilities import Probabilities

    probabilities = Probabilities(top, len(ids), 1)
    out: list[int] = []
    engine.generate(ids, 1, Sampling(seed=0, temperature=0.0), lambda new: out.extend(new) or False,
                    probabilities=probabilities)
    return probabilities.emitted(out[:1])[0]


def reference(model: str, a_path: Path, b_path: Path, out_path: Path, context: int) -> None:
    import moe_reference

    diff = compare(a_path, b_path)
    a_rows = {r["name"]: r for r in json.loads(a_path.read_text())["rows"]}
    engine, app = _engine(model, context)
    moe_reference.install()
    items = {p["name"]: p for p in prompts()}
    results = []
    for r in diff:
        if r["equal"]:
            results.append(r)
            continue
        item = items[r["name"]]
        prompt = app.prepare(body(item["prompt"], item["max_tokens"], 5), True).prompt
        ids = list(prompt) + list(a_rows[r["name"]]["token_ids"][:r["index"]])
        t0 = time.time()
        ref = _next_token_top(engine, ids)
        top = {int(i): float(v) for i, v in ref["top"]}
        a_id, b_id = r["a"]["id"], r["b"]["id"]
        la, lb = top.get(a_id), top.get(b_id)
        agrees = None if la is None and lb is None else ("A" if (lb is None or (la is not None and la > lb)) else "B")
        r = {**r, "reference": {"top": ref["top"][:5], "logprob_a_token": la, "logprob_b_token": lb,
                                "argmax": int(ref["id"]), "agrees_with": agrees, "seconds": time.time() - t0,
                                "prefix_tokens": len(ids)}}
        print(f"{r['name']:14s} referee: A's token lp {la}, B's token lp {lb}, argmax {ref['id']} -> agrees with "
              f"{agrees} ({time.time() - t0:.0f}s, {len(ids)} tokens)", flush=True)
        results.append(r)
        out_path.write_text(json.dumps({"a": str(a_path), "b": str(b_path), "rows": results}, indent=1))
    out_path.write_text(json.dumps({"a": str(a_path), "b": str(b_path), "rows": results}, indent=1))


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
        reference(args.model, args.a, args.b, args.out, args.context)
    else:
        selfcheck(args.model, args.layer)


if __name__ == "__main__":
    main()
