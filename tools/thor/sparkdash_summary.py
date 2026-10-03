"""tools/thor/sparkdash_summary.py: repeated sparkDash runs (DECODE-0 driver JSONs) -> per config, type and stream
count means with 95% Student-t CIs, pooled MTP acceptance and tokens a round, parity verdicts against a baseline config
and a reference table, plus a digest of the server's per-request "done" lines."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_T95_SMALL = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
              2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
              2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]

_DONE_TAG = "[tensorfold] done"

_DECODE_HEADER = ("| config | type | streams | n | aggregate tok/s (95% CI) | per request tok/s (95% CI) | TTFT ms | "
                  "acceptance | tokens/round | vs baseline | vs reference |")
_PREFILL_HEADER = ("| config | prompt | n | prefill tok/s (95% CI) | prompt tokens | TTFT ms | vs baseline | "
                   "vs reference |")


def t95(df: int) -> float:
    """Two-sided 95% Student-t critical value for df degrees of freedom."""
    if df <= 0:
        raise ValueError(f"df must be >= 1, got {df}")
    if df <= 30:
        return _T95_SMALL[df - 1]
    if df <= 40:
        return 2.021
    if df <= 60:
        return 2.000
    if df <= 120:
        return 1.980
    return 1.960


def ci95(values: list[float]) -> dict:
    """Mean with two-sided 95% t interval; n=0/1 keep the mean only."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "half": None, "lo": None, "hi": None}
    if n == 1:
        return {"n": 1, "mean": values[0], "sd": None, "half": None, "lo": None, "hi": None}
    mean = sum(values) / n
    sd = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5
    half = t95(n - 1) * sd / n ** 0.5
    return {"n": n, "mean": mean, "sd": sd, "half": half, "lo": mean - half, "hi": mean + half}


def verdict(cand: dict, base: dict, tol: float = 0.03) -> str:
    """Parity label for two ci95 dicts: within tol of the baseline mean, overlapping CIs, or clearly one-sided."""
    if cand["mean"] is None or base["mean"] is None:
        return "n/a"
    if abs(cand["mean"] - base["mean"]) <= tol * base["mean"]:
        return "within"
    if (cand["lo"] is not None and base["lo"] is not None
            and cand["lo"] <= base["hi"] and base["lo"] <= cand["hi"]):
        return "overlap"
    return "above" if cand["mean"] > base["mean"] else "below"


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _mean(vals):
    return sum(vals) / len(vals) if vals else None


def _split_label(label: str) -> tuple[str, str]:
    if "_" in label:
        cfg, rnd = label.rsplit("_", 1)
        return cfg, rnd
    return label, ""


def _decode_row(cfg: str, kind: str, conc, e: dict) -> dict:
    ok = e["ok"]
    servers = [(w.get("server") or {}) for w in ok]
    da = [(s["drafted"], s["accepted"]) for s in servers
          if _is_num(s.get("drafted")) and _is_num(s.get("accepted"))]
    d_sum = sum(d for _, d in da)
    gr = [(s["gen"] - s["requests"], s["rounds"]) for s in servers
          if _is_num(s.get("gen")) and _is_num(s.get("requests")) and _is_num(s.get("rounds")) and s["rounds"] > 0]
    return {
        "config": cfg, "type": kind, "conc": conc, "n": len(ok), "failed": e["failed"],
        "agg": ci95([w["aggregateDecodeTps"] for w in ok]),
        "per_req": ci95([w["meanDecodeTps"] for w in ok]),
        "ttft_ms": _mean([w["meanTtftMs"] for w in ok]),
        "tokens": _mean([w["totalCompletionTokens"] / conc for w in ok]),
        "acceptance": (sum(d for _, d in da) / sum(a for a, _ in da)) if d_sum else None,
        "tokens_per_round": (sum(g for g, _ in gr) / sum(r for _, r in gr)) if gr else None,
        "by_round": {rnd: _mean(aggs) for rnd, aggs in e["by_round"].items()},
    }


def _pct(cand_mean, base_mean):
    if cand_mean is None or not base_mean:
        return None
    return 100 * cand_mean / base_mean


def summarize(docs: list[dict], baseline: str | None = None, reference: dict | None = None) -> dict:
    """Aggregate driver docs into decode rows keyed (config, type, conc) and prefill rows keyed (config, target)."""
    dec: dict = {}
    dec_order: dict = {}
    for doc in docs:
        cfg, rnd = _split_label(doc.get("label", ""))
        for w in doc.get("waves", []):
            kind, conc = w["type"], w["concurrency"]
            e = dec.setdefault((cfg, kind, conc), {"ok": [], "failed": 0, "by_round": {}})
            dec_order.setdefault(cfg, {}).setdefault(kind)
            if w.get("streamsOk") == conc and not w.get("error"):
                e["ok"].append(w)
                e["by_round"].setdefault(rnd, []).append(w["aggregateDecodeTps"])
            else:
                e["failed"] += 1
    decode = []
    for cfg, types in dec_order.items():
        for kind in types:
            for conc in sorted(c for (c2, k2, c) in dec if c2 == cfg and k2 == kind):
                decode.append(_decode_row(cfg, kind, conc, dec[(cfg, kind, conc)]))
    base_rows = {(r["type"], r["conc"]): r for r in decode if baseline is not None and r["config"] == baseline}
    for r in decode:
        if baseline is not None and r["config"] != baseline:
            base = base_rows.get((r["type"], r["conc"]))
            if base is not None:
                r["vs_baseline"] = {"pct": _pct(r["agg"]["mean"], base["agg"]["mean"]),
                                    "verdict": verdict(r["agg"], base["agg"])}
        if reference:
            ref = reference.get("decode", {}).get(r["type"], {}).get(str(r["conc"]))
            if ref is not None:
                r["vs_reference_pct"] = _pct(r["agg"]["mean"], ref)

    pre: dict = {}
    pre_order: dict = {}
    for doc in docs:
        cfg, _ = _split_label(doc.get("label", ""))
        for w in doc.get("prefill", []):
            e = pre.setdefault((cfg, w["targetTokens"]), {"ok": [], "failed": 0})
            pre_order.setdefault(cfg, {})
            if not w.get("error") and (w.get("prefillTps") or 0) > 0:
                e["ok"].append(w)
            else:
                e["failed"] += 1
    prefill = []
    for cfg in pre_order:
        for target in sorted(t for (c, t) in pre if c == cfg):
            e = pre[(cfg, target)]
            ok = e["ok"]
            prefill.append({
                "config": cfg, "target": target, "n": len(ok), "failed": e["failed"],
                "tps": ci95([w["prefillTps"] for w in ok]),
                "prompt_tokens": _mean([w["promptTokens"] for w in ok]),
                "ttft_ms": _mean([w["ttftMs"] for w in ok]),
            })
    base_pre = {r["target"]: r for r in prefill if baseline is not None and r["config"] == baseline}
    for r in prefill:
        if baseline is not None and r["config"] != baseline:
            base = base_pre.get(r["target"])
            if base is not None:
                r["vs_baseline"] = {"pct": _pct(r["tps"]["mean"], base["tps"]["mean"]),
                                    "verdict": verdict(r["tps"], base["tps"])}
        if reference:
            ref = reference.get("prefill", {}).get(str(r["target"]))
            if ref is not None:
                r["vs_reference_pct"] = _pct(r["tps"]["mean"], ref)
    return {"decode": decode, "prefill": prefill}


def _sep(header: str) -> str:
    return "|---" * len(header.strip("|").split(" | ")) + "|"


def _n_cell(n: int, failed: int) -> str:
    return str(n) + (f" ({failed} failed)" if failed else "")


def _ci_cell(ci: dict) -> str:
    if ci["mean"] is None:
        return "-"
    if ci["lo"] is not None:
        return f'{ci["mean"]:.2f} [{ci["lo"]:.2f}, {ci["hi"]:.2f}]'
    return f'{ci["mean"]:.2f}'


def _fmt(v, spec: str) -> str:
    return "-" if v is None else format(v, spec)


def _vb_cell(row: dict) -> str:
    vb = row.get("vs_baseline")
    if not vb or vb["pct"] is None:
        return "-"
    return f'{vb["pct"]:.1f}% {vb["verdict"]}'


def _vr_cell(row: dict) -> str:
    return "-" if row.get("vs_reference_pct") is None else f'{row["vs_reference_pct"]:.1f}%'


def markdown(summary: dict) -> str:
    """Render the summary as a decode and a prefill markdown table, every line newline-terminated."""
    lines = [_DECODE_HEADER, _sep(_DECODE_HEADER)]
    for r in summary["decode"]:
        cells = [str(r["config"]), str(r["type"]), str(r["conc"]), _n_cell(r["n"], r["failed"]),
                 _ci_cell(r["agg"]), _ci_cell(r["per_req"]), _fmt(r["ttft_ms"], ".0f"),
                 _fmt(r["acceptance"], ".3f"), _fmt(r["tokens_per_round"], ".2f"),
                 _vb_cell(r), _vr_cell(r)]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(_PREFILL_HEADER)
    lines.append(_sep(_PREFILL_HEADER))
    for r in summary["prefill"]:
        cells = [str(r["config"]), str(r["target"]), _n_cell(r["n"], r["failed"]), _ci_cell(r["tps"]),
                 _fmt(r["prompt_tokens"], ".0f"), _fmt(r["ttft_ms"], ".0f"), _vb_cell(r), _vr_cell(r)]
        lines.append("| " + " | ".join(cells) + " |")
    return "".join(line + "\n" for line in lines)


def digest_done(lines: list[str], min_tokens: int = 390) -> dict:
    """Pool the server's '[tensorfold] done' lines with tokens >= min_tokens."""
    kept = []
    for line in lines:
        if _DONE_TAG not in line:
            continue
        m = re.search(r"tokens=(\d+)", line)
        if m and int(m.group(1)) >= min_tokens:
            kept.append(line)
    profiled = [ln for ln in kept if re.search(r"ms/round=", ln)]

    def _mean_over(haystack: list[str], pattern: str):
        vals = [float(m.group(1)) for ln in haystack for m in [re.search(pattern, ln)] if m]
        return _mean(vals)

    pairs = []
    for ln in kept:
        tm, rm = re.search(r"tokens=(\d+)", ln), re.search(r"rounds=(\d+)", ln)
        if tm and rm:
            pairs.append((int(tm.group(1)), int(rm.group(1))))
    rounds_sum = sum(r for _, r in pairs)
    accs = []
    for ln in kept:
        m = re.search(r"accepted=(\d+)/(\d+)", ln)
        if m:
            accs.append((int(m.group(1)), int(m.group(2))))
    d_sum = sum(d for _, d in accs)
    return {
        "requests": len(kept),
        "profiled": len(profiled),
        "ms_round": _mean_over(profiled, r"ms/round=(\d+(?:\.\d+)?)"),
        "forward_ms": _mean_over(profiled, r" forward=(\d+(?:\.\d+)?)"),
        "draft_ms": _mean_over(profiled, r" draft=(\d+(?:\.\d+)?)"),
        "post_ms": _mean_over(profiled, r" post=(\d+(?:\.\d+)?)"),
        "rows": _mean_over(profiled, r" rows=(\d+(?:\.\d+)?)"),
        "tokens_per_round": (sum(t - 1 for t, _ in pairs) / rounds_sum) if rounds_sum else None,
        "acceptance": (sum(a for a, _ in accs) / d_sum) if d_sum else None,
        "server_tok_s": _mean_over(kept, r"tok/s=(\d+(?:\.\d+)?)"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", help="directory holding driver JSONs and serve_<tag>.log files")
    parser.add_argument("--glob", default="d0_*.json", help="pattern of driver JSONs to load")
    parser.add_argument("--json", dest="json_path", help="write the summary JSON here")
    parser.add_argument("--markdown", dest="markdown_path", help="write the markdown tables here")
    parser.add_argument("--baseline", help="config name the others are compared against")
    parser.add_argument("--reference", help="JSON file with reference decode/prefill tables")
    parser.add_argument("--min-tokens", type=int, default=390, help="minimum tokens for a done line to count")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    docs = [json.loads(path.read_text()) for path in sorted(out_dir.glob(args.glob), key=lambda pth: pth.name)]
    reference = json.loads(Path(args.reference).read_text()) if args.reference else None
    summary = summarize(docs, baseline=args.baseline, reference=reference)
    servers = {}
    for log in sorted(out_dir.glob("serve_*.log"), key=lambda pth: pth.name):
        servers[log.stem[len("serve_"):]] = digest_done(log.read_text().splitlines(), min_tokens=args.min_tokens)
    summary["servers"] = servers
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(summary, indent=1) + "\n")
    md = markdown(summary)
    if args.markdown_path:
        Path(args.markdown_path).write_text(md)
    print(md, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
