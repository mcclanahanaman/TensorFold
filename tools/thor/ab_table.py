"""A/B result tables for the serve_ab.py and natural_prefill.py reports: per-run metrics, per-config means and
ratios, same-reply-token checks, and a one-run ledger of the headline numbers. When a report carries
decode.by_streams the decode metrics are emitted per stream count, and each follow-up target's first-turn
TTFT becomes a "first turn ... s" metric.

Reads DIR/ab_<config>_<round>.json (serve_ab.py) and DIR/nat_<config>_<round>.json (natural_prefill.py); a
missing file simply contributes no metrics for that run. Writes the {'summary': ..., 'same_tokens': ...} JSON,
the markdown table and the one-run ledger to the files given on the command line, and prints the table.

    python tools/thor/ab_table.py DIR [--configs stock,off,on] [--rounds 1,2] [--baseline stock]
        [--json FILE] [--markdown FILE] [--ledger FILE] [--ledger-run CONFIG:ROUND]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

LEDGER_KEYS = [("prefill_8k", "prefill 8k"), ("prefill_32k", "prefill 32k"), ("prefill_128k", "prefill 128k"),
               ("decode_1", "decode 1 tok/s"), ("decode_3", "decode 3 wall tok/s"),
               ("users_4", "decode 4 wall tok/s"), ("followup_s", "follow-up 60k s")]


def load(path: Path):
    """The JSON report at path, or None when the file is missing."""
    return json.loads(path.read_text()) if path.exists() else None


def run_metrics(ab, nat):
    """The ordered {metric: value} of one run: the serve_ab.py report ab and the natural_prefill.py report nat."""
    m = {}
    if ab:
        prefill = ab.get("prefill") or []
        targets = []
        for row in prefill:
            if row["target"] not in targets:
                targets.append(row["target"])
        for t in targets:
            vals = [r["tok_s"] for r in prefill if r["target"] == t]
            m[f"prefill {t // 1024}k"] = st.mean(vals)
            m[f"prefill {t // 1024}k first"] = vals[0]
        followups = ab.get("followups") or []
        for t in sorted({f["target"] for f in followups}):
            rows_t = [f for f in followups if f["target"] == t]
            first = [f["first"]["ttft_s"] for f in rows_t if f["first"].get("ttft_s") is not None]
            if first:
                m[f"first turn {t // 1024}k s"] = st.mean(first)
            nxt = [f["next"] for f in rows_t]
            m[f"follow-up {t // 1024}k s"] = st.mean(x["ttft_s"] for x in nxt)
            m[f"follow-up {t // 1024}k filled"] = st.mean(x["prompt_tokens"] - x["cached"] for x in nxt)
            m[f"follow-up {t // 1024}k resumed at"] = st.mean(x["cached"] for x in nxt)
        d = ab.get("decode") or {}
        one = d.get("one")
        if one is not None:
            m["decode 1 tok/s"] = one.get("server_decode_tok_s")
            if one.get("acceptance") is not None:
                m["decode 1 MTP acceptance"] = one["acceptance"]
        by_streams = d.get("by_streams")
        if by_streams is not None:
            for k in sorted(by_streams, key=int):
                n, e = int(k), by_streams[k]
                m[f"decode {n} wall tok/s"] = e["wall_tok_s"]
                m[f"decode {n} server sum"] = e["server_sum_tok_s"]
                acc = [s.get("acceptance") for s in e.get("streams") or []
                       if s.get("acceptance") is not None]
                if acc:
                    m[f"decode {n} MTP acceptance"] = st.mean(acc)
        else:
            many = d.get("many")
            if many is not None:
                m["decode 3 wall tok/s"] = many.get("wall_tok_s")
                m["decode 3 server sum"] = many.get("server_sum_tok_s")
                acc = [s.get("acceptance") for s in many.get("streams") or []
                       if s.get("acceptance") is not None]
                if acc:
                    m["decode 3 MTP acceptance"] = st.mean(acc)
    if nat:
        tokens = nat.get("tokens", 32768)
        for kind in ("cold", "warm"):
            vals = [r["tok_s"] for r in nat.get("runs") or []
                    if r.get("kind") == kind and r.get("tok_s") is not None]
            if vals:
                m[f"natural {tokens // 1024}k {kind}"] = st.mean(vals)
    return m


def run_shas(ab):
    """A report's reply token hashes in report order: prefill rows, first turns, decode.one, decode.many streams."""
    if not ab:
        return None
    d = ab.get("decode") or {}
    by = d.get("by_streams")
    streams = ([s.get("token_sha") for k in sorted(by, key=int) for s in by[k].get("streams") or []]
               if by is not None
               else [s.get("token_sha") for s in (d.get("many") or {}).get("streams") or []])
    return ([r.get("token_sha") for r in ab.get("prefill") or []]
            + [f["first"].get("token_sha") for f in ab.get("followups") or []]
            + [(d.get("one") or {}).get("token_sha")] + streams)


def build_summary(rows, configs, rounds, baseline):
    """summary: per metric the by_run values over all runs, the per-config means and the ratios vs the baseline."""
    keys = []
    for c in configs:
        for r in rounds:
            keys += [k for k in rows.get((c, r), {}) if k not in keys]
    summary = {}
    for k in keys:
        by_run = {f"{c} r{r}": rows.get((c, r), {}).get(k) for c in configs for r in rounds}
        mean = {}
        for c in configs:
            vals = [v for r in rounds if (v := rows.get((c, r), {}).get(k)) is not None]
            mean[c] = st.mean(vals) if vals else None
        base = mean.get(baseline)
        ratio = {c: (mean[c] / base if base is not None and base != 0 and mean[c] is not None else None)
                 for c in configs if c != baseline}
        summary[k] = {"by_run": by_run, "mean": mean, "ratio": ratio}
    return summary


def same_tokens(hashes, configs, rounds, baseline):
    """Same-reply-token checks: each non-baseline config vs the baseline per round, then the first two rounds."""
    same = {}
    for c in configs:
        if c == baseline:
            continue
        for r in rounds:
            a, b = hashes.get((baseline, r)), hashes.get((c, r))
            if a is not None and b is not None:
                same[f"{c} r{r} vs {baseline} r{r}"] = f"{sum(x == y for x, y in zip(a, b))}/{len(a)}"
    if len(rounds) >= 2:
        r1, r2 = rounds[0], rounds[1]
        for c in configs:
            a, b = hashes.get((c, r1)), hashes.get((c, r2))
            if a is not None and b is not None:
                same[f"{c} r{r1} vs r{r2}"] = f"{sum(x == y for x, y in zip(a, b))}/{len(a)}"
    return same


def fmt(v):
    """One markdown cell: null as '-', three decimals when |v| < 10, whole tok/s otherwise."""
    return "-" if v is None else f"{v:.3f}" if abs(v) < 10 else f"{v:.0f}"


def markdown_table(summary, keys, configs, rounds, baseline):
    """The markdown table: a column per run (configs then rounds), then a ratio column per non-baseline config."""
    cols = [f"{c} r{r}" for c in configs for r in rounds] + [f"{c}/{baseline}" for c in configs if c != baseline]
    lines = ["| metric | " + " | ".join(cols) + " |",
             "|" + "|".join(["-" * 10] * (len(cols) + 1)) + "|"]
    for k in keys:
        s = summary[k]
        cells = [fmt(s["by_run"][f"{c} r{r}"]) for c in configs for r in rounds]
        cells += [fmt(s["ratio"].get(c)) for c in configs if c != baseline]
        lines.append("| " + k + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ledger_run(arg, configs, rounds):
    """The (config, round) the --ledger is taken from: CONFIG:ROUND, default the last config and first round."""
    if arg:
        c, r = arg.split(":")
        return c, int(r)
    return configs[-1], rounds[0]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="A/B result tables for serve_ab.py and natural_prefill.py reports.")
    p.add_argument("dir", type=Path, help="directory with the ab_*/nat_* <config>_<round>.json reports")
    p.add_argument("--configs", default="stock,off,on", help="comma-separated configs, in table order")
    p.add_argument("--rounds", default="1,2", help="comma-separated rounds, in table order")
    p.add_argument("--baseline", default="stock", help="config the ratios are taken against")
    p.add_argument("--json", type=Path, help="write {'summary': ..., 'same_tokens': ...} here")
    p.add_argument("--markdown", type=Path, help="write the markdown table here")
    p.add_argument("--ledger", type=Path, help="write the --ledger-run run's headline metrics here")
    p.add_argument("--ledger-run", help="CONFIG:ROUND the ledger is taken from (default: last config, first round)")
    args = p.parse_args(argv)

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    rounds = [int(r) for r in args.rounds.split(",") if r.strip()]
    rows, hashes = {}, {}
    for c in configs:
        for r in rounds:
            ab = load(args.dir / f"ab_{c}_{r}.json")
            rows[(c, r)] = run_metrics(ab, load(args.dir / f"nat_{c}_{r}.json"))
            hashes[(c, r)] = run_shas(ab)
    summary = build_summary(rows, configs, rounds, args.baseline)
    same = same_tokens(hashes, configs, rounds, args.baseline)
    text = markdown_table(summary, list(summary), configs, rounds, args.baseline)
    text += "\nsame reply tokens: " + "; ".join(f"{k} {v}" for k, v in same.items())
    print(text)
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "same_tokens": same}, indent=2))
    if args.markdown:
        args.markdown.write_text(text + "\n")
    if args.ledger:
        c, r = ledger_run(args.ledger_run, configs, rounds)
        m = rows.get((c, r), {})
        args.ledger.write_text(json.dumps({key: round(m[name], 3) for key, name in LEDGER_KEYS
                                           if m.get(name) is not None}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
