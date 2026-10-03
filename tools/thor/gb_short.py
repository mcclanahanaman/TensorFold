"""The short greedy G-B probe of THOR-DEC-1: sparkDash prompts sent one at a time at temperature 0, so every
decode forward is one stream within the QSA budget and the one-stream decode graphs replay. ``collect`` sends
the prompts to an OpenAI-compatible server and writes the outputs; ``compare`` checks two collections byte
for byte.

    python tools/thor/gb_short.py collect URL OUT.json [--sparkdash PATH] [--types prose,code,structured,json]
        [--per-type 4] [--max-tokens 400,1200] [--label L]
    python tools/thor/gb_short.py compare A.json B.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_sparkdash(path: Path):
    """The stdlib sparkDash port, loaded the way decode_phases.py loads it."""
    spec = importlib.util.spec_from_file_location("sparkdash_port", path)
    sd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sd)
    return sd


def _http_json(url: str, body: dict | None, timeout: float) -> dict:
    """One HTTP round trip: a POST of ``body`` as JSON when given, else a GET; the response's JSON back."""
    data, headers, method = None, {}, "GET"
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def collect(args) -> int:
    """Send each sparkDash prompt once per max_tokens value, one request at a time, and write the outputs."""
    base = args.url.rstrip("/")
    types = [t for t in args.types.split(",") if t]
    max_tokens = [int(x) for x in args.max_tokens.split(",") if x.strip()]
    sd = _load_sparkdash(args.sparkdash)
    model = _http_json(base + "/v1/models", None, args.timeout)["data"][0]["id"]
    items: list = []
    report = {"label": args.label, "url": base, "model": model,
              "protocol": {"types": types, "per_type": args.per_type, "max_tokens": max_tokens,
                           "temperature": 0, "sequential": True},
              "items": items}

    def flush() -> None:
        args.out.write_text(json.dumps(report), encoding="utf-8")

    for kind in types:
        prompts = sd.pick_decode_bench_prompts(args.per_type, kind)
        for i, prompt in enumerate(prompts):
            for n in max_tokens:
                body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0, "top_p": 1, "max_tokens": n, "stream": False}
                body = sd.apply_thinking_flags(body, model, think=False)
                t0 = time.perf_counter()
                resp = _http_json(base + "/v1/chat/completions", body, args.timeout)
                seconds = round(time.perf_counter() - t0, 3)
                choice = (resp.get("choices") or [{}])[0]
                usage = resp.get("usage") or {}
                item = {"id": f"{kind}#{i}@{n}", "type": kind, "index": i, "max_tokens": n, "prompt": prompt,
                        "text": (choice.get("message") or {}).get("content") or "",
                        "finish_reason": choice.get("finish_reason"),
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"), "seconds": seconds}
                items.append(item)
                flush()
                print(f"[gb] {item['id']}: completion_tokens {item['completion_tokens']}, finish "
                      f"{item['finish_reason']}, {seconds:.3f} s", flush=True)
    flush()
    return 0


def compare(args) -> int:
    """Two collections byte for byte: per-id text diffs with the first differing character, and the id sets."""
    a = json.loads(args.a.read_text(encoding="utf-8"))
    b = json.loads(args.b.read_text(encoding="utf-8"))
    items_a, items_b = a.get("items", []), b.get("items", [])
    by_id = {it.get("id"): it for it in items_b}
    ids_a = {it.get("id") for it in items_a}
    ok, identical = True, 0
    for it in items_a:
        iid = it.get("id")
        other = by_id.get(iid)
        if other is None:
            print(f"missing in B: {iid}")
            ok = False
            continue
        ta, tb = it.get("text") or "", other.get("text") or ""
        if ta == tb:
            identical += 1
            continue
        j = min(len(ta), len(tb))
        for k in range(j):
            if ta[k] != tb[k]:
                j = k
                break
        print(f"{iid}: DIFF at char {j} (completion tokens {it.get('completion_tokens')} vs "
              f"{other.get('completion_tokens')})")
        print(f"    A: {ta[j:j + 40]!r}")
        print(f"    B: {tb[j:j + 40]!r}")
        ok = False
    for it in items_b:
        if it.get("id") not in ids_a:
            print(f"missing in A: {it.get('id')}")
            ok = False
    print(f"identical {identical}/{len(items_a)}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="The short greedy G-B probe of THOR-DEC-1.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("collect", help="send the prompts one at a time and write the collection")
    p.add_argument("url", help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8000")
    p.add_argument("out", type=Path, help="collection JSON to write (rewritten after every item)")
    p.add_argument("--sparkdash", type=Path, default=HERE / "_ext" / "decode0" / "sparkdash_port.py",
                   help="the stdlib sparkDash port (pick_decode_bench_prompts, apply_thinking_flags)")
    p.add_argument("--types", default="prose,code,structured,json", help="comma list of prompt types, in order")
    p.add_argument("--per-type", type=int, default=4, help="prompts per type")
    p.add_argument("--max-tokens", default="400,1200", help="comma list of completion budgets, in order")
    p.add_argument("--label", default="", help="label stored in the collection")
    p.add_argument("--timeout", type=float, default=600.0, help="per-request timeout in seconds")
    p.set_defaults(func=collect)

    p = sub.add_parser("compare", help="check two collections byte for byte")
    p.add_argument("a", type=Path, help="collection A")
    p.add_argument("b", type=Path, help="collection B")
    p.set_defaults(func=compare)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
