"""tools/thor/decode_bytes.py: a decode round's weight bytes from a checkpoint inventory (DECODE-0's inv.py JSON) and
measured MTP statistics (sparkdash_summary.py JSON): T-025's verify-pass model (unique routed experts under independent
top-k routing plus the fixed dense, hc, head and router bytes) extended with the MTP draft steps, against the DRAM floor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BW_GBPS = 256.0

CLASSES = ("expert", "mtp_expert", "router", "hc", "head", "mtp", "dense", "skip")

_HEADER = ("| config | type | streams | tokens/round | acceptance | drafts | verify rows | unique experts "
           "| expert GB | fixed GB | draft GB | total GB | floor ms | measured ms/round | floor % |")


def classify(name: str) -> str:
    if ("vision_tower" in name or ".visual." in name or name.startswith("visual.")
            or "embed_tokens" in name or "ngram_embedding" in name):
        return "skip"
    if name.startswith("mtp.") or ".mtp." in name:
        return "mtp_expert" if (".experts." in name or ".switch_mlp." in name) else "mtp"
    if "lm_head" in name:
        return "head"
    if ".experts." in name or ".switch_mlp." in name:
        return "expert"
    if ".mlp.gate." in name:
        return "router"
    if "_hyper_connection." in name or "hyper_connection_mixer" in name:
        return "hc"
    if ".layers." in name:
        return "dense"
    return "skip"


def classes(inventory: dict) -> dict:
    sums = dict.fromkeys(CLASSES, 0)
    for group in inventory["groups"]:
        sums[classify(group[0])] += group[3]
    return sums


def model_dims(inventory: dict) -> dict:
    cfg = inventory.get("config") or {}
    t = cfg["text_config"] if isinstance(cfg.get("text_config"), dict) else cfg
    return {"layers": t.get("num_hidden_layers", 48), "experts": t.get("num_experts", 512),
            "topk": t.get("num_experts_per_tok", 10), "hidden": t.get("hidden_size", 2560),
            "vocab": t.get("vocab_size", 248320)}


def draft_head_bytes(ids: int, hidden: int, bits: int = 4, group: int = 32) -> float:
    return ids * hidden * (bits / 8 + 4 / group)


def round_bytes(cls: dict, dims: dict, tpr: float, acc: float | None, users: int = 1,
                draft_head: float = 0.0, bw_gbps: float = BW_GBPS) -> dict:
    e, k, layers = dims["experts"], dims["topk"], dims["layers"]
    per_expert = cls["expert"] / (layers * e)
    mtp_per_expert = cls["mtp_expert"] / e
    drafts = (tpr - 1) / acc if (acc is not None and acc > 0 and tpr > 1) else 0.0
    rows = users * (1 + drafts)
    unique = e * (1 - (1 - k / e) ** rows)
    expert_b = unique * layers * per_expert
    fixed_b = cls["dense"] + cls["hc"] + cls["head"] + cls["router"]
    verify_b = expert_b + fixed_b
    step_unique = e * (1 - (1 - k / e) ** users)
    step_b = cls["mtp"] + step_unique * mtp_per_expert + draft_head
    draft_b = drafts * step_b
    total_b = verify_b + draft_b
    floor = lambda b: b / (bw_gbps * 1e9) * 1e3
    return {"drafts": float(drafts), "rows": float(rows), "unique_experts": float(unique),
            "expert_gb": expert_b / 1e9, "fixed_gb": fixed_b / 1e9, "verify_gb": verify_b / 1e9,
            "draft_step_gb": step_b / 1e9, "draft_gb": draft_b / 1e9, "total_gb": total_b / 1e9,
            "floor_ms": floor(total_b), "verify_floor_ms": floor(verify_b)}


def table(summary: dict, inventories: dict, draft_ids: int = 79591, bw_gbps: float = BW_GBPS) -> list[dict]:
    out = []
    for row in summary["decode"]:
        config = row["config"]
        if config not in inventories or row.get("tokens_per_round") is None:
            continue
        inv = inventories[config]
        dims = model_dims(inv)
        tpr = row["tokens_per_round"]
        r = round_bytes(classes(inv), dims, tpr, row.get("acceptance"), users=row["conc"],
                        draft_head=draft_head_bytes(draft_ids, dims["hidden"]), bw_gbps=bw_gbps)
        per = (row.get("per_req") or {}).get("mean")
        measured_ms = 1000 * tpr / per if per else None
        entry = {"config": config, "type": row["type"], "conc": row["conc"], "tpr": tpr,
                 "acc": row.get("acceptance")}
        entry.update(r)
        entry["measured_ms"] = measured_ms
        entry["floor_pct"] = 100 * r["floor_ms"] / measured_ms if measured_ms else None
        out.append(entry)
    return out


def _cell(value, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def markdown(rows: list[dict]) -> str:
    lines = [_HEADER, "|" + "---|" * 14 + "---|"]
    for r in rows:
        cells = [str(r["config"]), str(r["type"]), str(r["conc"]), _cell(r["tpr"], ".2f"),
                 _cell(r["acc"], ".3f"), _cell(r["drafts"], ".2f"), _cell(r["rows"], ".2f"),
                 _cell(r["unique_experts"], ".1f"), _cell(r["expert_gb"], ".2f"),
                 _cell(r["fixed_gb"], ".2f"), _cell(r["draft_gb"], ".2f"), _cell(r["total_gb"], ".2f"),
                 _cell(r["floor_ms"], ".1f"), _cell(r["measured_ms"], ".1f"),
                 "-" if r["floor_pct"] is None else f"{r['floor_pct']:.0f}%"]
        lines.append("| " + " | ".join(cells) + " |")
    return "".join(line + "\n" for line in lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Decode-round weight bytes vs the DRAM floor, from inventories "
                                            "and measured MTP statistics.")
    p.add_argument("summary", help="sparkdash_summary.py JSON path")
    p.add_argument("--inv", action="append", default=[], metavar="NAME=PATH",
                   help="config NAME with its inventory JSON PATH (repeatable)")
    p.add_argument("--draft-ids", type=int, default=79591, help="ids in the draft vocabulary (TensorFold cuda/draft_vocab.txt: 79591)")
    p.add_argument("--bw", type=float, default=BW_GBPS, help="DRAM bandwidth in GB/s (default 256)")
    p.add_argument("--json", metavar="PATH", help="write the full result as JSON to PATH")
    p.add_argument("--markdown", metavar="PATH", help="write the markdown table to PATH")
    args = p.parse_args(argv)

    summary = json.loads(Path(args.summary).read_text())
    inventories = {}
    for item in args.inv:
        name, sep, path = item.partition("=")
        if not sep:
            p.error(f"--inv expects NAME=PATH, got {item!r}")
        inventories[name] = json.loads(Path(path).read_text())

    rows = table(summary, inventories, draft_ids=args.draft_ids, bw_gbps=args.bw)
    md = markdown(rows)
    if args.json:
        payload = {"bw_gbps": args.bw, "draft_ids": args.draft_ids,
                   "classes": {n: classes(inv) for n, inv in inventories.items()},
                   "dims": {n: model_dims(inv) for n, inv in inventories.items()}, "rows": rows}
        Path(args.json).write_text(json.dumps(payload, indent=1))
    if args.markdown:
        Path(args.markdown).write_text(md)
    sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
