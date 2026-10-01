"""Natural-text prefill on a running TensorFold server, warm and with the n-gram table cold (FR-OPS-17 gate KPI).

    python tools/thor/natural_prefill.py BASE_URL --model qwen3.8-flash-next --tokens 32768 --out FILE
        [--drop-table SNAPSHOT_DIR] [--snapshot SNAPSHOT_DIR] [--repeat N] [--seed S]

Each request is a new prompt (nothing resumes from a kept prompt end): prose and code from this tree, the source of
``prefill_profile.py --natural``, cut to about ``--tokens`` tokens by the model's tokenizer (tokenizer.json in the
snapshot; the server's own count is reported), one token out, streamed. TTFT is the arrival of the first chunk after
the server's opening role chunk; prefill tok/s is the server's prompt token count (less any cached) over TTFT.

``--drop-table SNAPSHOT_DIR``: each round is a cold request, then a warm one. Before the cold request the n-gram
table's byte ranges (the ``ngram_embedding`` tensors of the snapshot's safetensors files) get POSIX_FADV_DONTNEED, which
needs no privileges but drops only the pages no process maps: the server maps the pages its lookups touched (and the
kernel maps up to 64 KiB around each fault), so the table goes fully cold only on a server that has served little.
On a fresh server the first cold request is the cleanest; ``residency_before`` (mincore over the ranges, mapped
read-only in this process) says how cold each one was: 0.47-0.29 of the table stayed resident after one and two
32k prompts in the tool's test. Before each warm request the ranges are read back into the page cache.

Every request is a new text: the seed (``--seed``, else the clock) advances a request. Use the same ``--seed`` on
fresh servers to compare them on the same prompts, never twice against one server (a prompt it has kept resumes).
Standard library only, plus ``tokenizers`` (or ``transformers``).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import random
import statistics
import struct
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PAGE = mmap.PAGESIZE
READ = 16 << 20                 # bytes a warm-up read
OVERHEAD = 12                   # the chat template's tokens around one user message, thinking off (Flash Next's)


# ----------------------------------------------------------------------------------------------------- the table

def table_ranges(snapshot: Path) -> list[tuple[str, int, int]]:
    """(file, first byte, length) of each n-gram table tensor in the snapshot's safetensors files."""

    out = []
    for path in sorted(snapshot.glob("*.safetensors")):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        for name, info in header.items():
            if isinstance(info, dict) and "ngram_embedding" in name and "data_offsets" in info:
                begin, end = info["data_offsets"]
                out.append((str(path), 8 + n + int(begin), int(end) - int(begin)))
    return out


def drop(ranges) -> float:
    """POSIX_FADV_DONTNEED on every range: the pages no process maps leave the page cache."""

    t = time.perf_counter()
    for path, at, n in ranges:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, at, n, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    return time.perf_counter() - t


def warm(ranges, workers: int = 8) -> float:
    """Read every range once, READ bytes a read on ``workers`` threads, so the pages are in the page cache."""

    spans = [(path, at + o, min(READ, n - o)) for path, at, n in ranges for o in range(0, n, READ)]

    def read(span) -> None:
        path, at, n = span
        fd = os.open(path, os.O_RDONLY)
        try:
            os.pread(fd, n, at)
        finally:
            os.close(fd)

    t = time.perf_counter()
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(read, spans))
    return time.perf_counter() - t


def residency(ranges) -> float | None:
    """The fraction of the ranges' pages in the page cache (mincore on a read-only map of each); None if it fails."""

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ctypes.c_long)
        libc.munmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        libc.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
        have = total = 0
        for path, at, n in ranges:
            start = at - at % PAGE
            size = n + (at - start)
            fd = os.open(path, os.O_RDONLY)
            try:
                addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, start)
                if addr in (None, ctypes.c_void_p(-1).value):
                    return None
                try:
                    pages = -(-size // PAGE)
                    vec = (ctypes.c_ubyte * pages)()
                    if libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec) != 0:
                        return None
                    have += pages - bytes(vec).count(b"\x00")       # a byte's low bit: the page is resident
                    total += pages
                finally:
                    libc.munmap(ctypes.c_void_p(addr), ctypes.c_size_t(size))
            finally:
                os.close(fd)
        return have / total if total else None
    except (OSError, AttributeError, ValueError):
        return None


# ----------------------------------------------------------------------------------------------------- the prompt

def corpus() -> str:
    """``prefill_profile.py --natural``'s source: this tree's docs, then its sources (the installed package's, if this
    file sits outside a checkout)."""

    here = Path(__file__).resolve()
    root = here.parents[2] if len(here.parents) > 2 else here.parent
    files = sorted(root.glob("docs/**/*.md")) + sorted(root.glob("*.md")) + sorted(root.glob("src/**/*.py"))
    if not files:
        import importlib.util

        spec = importlib.util.find_spec("tensorfold")
        if spec is None or not spec.submodule_search_locations:
            raise SystemExit("no text source: run from a TensorFold checkout or with tensorfold installed")
        files = sorted(Path(list(spec.submodule_search_locations)[0]).glob("**/*.py"))
    return "\n\n".join(f.read_text(errors="ignore") for f in files)


def tokenizer(snapshot: Path | None):
    """encode(text) -> ids and decode(ids) -> text from the snapshot's tokenizer, or None."""

    if snapshot is None:
        return None
    try:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
        return (lambda s: tok.encode(s, add_special_tokens=False).ids), tok.decode
    except ImportError:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
        return (lambda s: tok.encode(s, add_special_tokens=False)), tok.decode


def prompt(text: str, seed: int, tokens: int, tok) -> tuple[str, int | None]:
    """A new user message of about ``tokens`` tokens with the template: its text, and its own token count."""

    head = f"Run {seed}. Summarize the following.\n\n"
    chars = tokens * 6
    start = random.Random(seed).randrange(max(1, len(text) - chars))
    body = (text + text)[start:start + chars]
    if tok is None:                                  # no tokenizer: about 2.75 characters a token in this source
        return head + body[:int((tokens - OVERHEAD) * 2.75)], None
    encode, decode = tok
    room = tokens - OVERHEAD - len(encode(head))
    content = head + decode(encode(body)[:room])
    return content, len(encode(content)) + OVERHEAD


# ----------------------------------------------------------------------------------------------------- a request

def request(base: str, model: str, content: str, timeout: float) -> dict:
    """One streamed chat request, one token out: TTFT (first chunk after the role chunk), the server's usage."""

    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 1, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Accept": "text/event-stream"})
    t0 = time.perf_counter()
    first = None
    usage, stats, text = {}, {}, ""
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(f"the server stream ended with an error: {event['error']}")
            choice = (event.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            opening = set(delta) <= {"role"} and choice.get("finish_reason") is None and "usage" not in event
            if first is None and not opening:
                first = time.perf_counter() - t0
            text += delta.get("content") or delta.get("reasoning_content") or ""
            usage = event.get("usage") or usage
            stats = event.get("tensorfold") or stats
    total = time.perf_counter() - t0
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or stats.get("cached") or 0)
    ttft = first if first is not None else total
    return {"prompt_tokens": prompt_tokens, "cached": cached, "ttft_s": round(ttft, 3), "total_s": round(total, 3),
            "tok_s": round((prompt_tokens - cached) / ttft, 1) if prompt_tokens and ttft > 0 else None,
            "server_prefill_s": stats.get("prefill_s"), "first_text": text[:40]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--drop-table", type=Path, default=None, metavar="SNAPSHOT_DIR")
    ap.add_argument("--snapshot", type=Path, default=None, help="the tokenizer's snapshot (default: --drop-table's)")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None, help="the first prompt's seed (default: the clock)")
    ap.add_argument("--timeout", type=float, default=3600.0)
    args = ap.parse_args()
    base = args.base.rstrip("/")
    snapshot = args.snapshot or args.drop_table
    tok = tokenizer(snapshot)
    text = corpus()
    ranges = table_ranges(args.drop_table) if args.drop_table else []
    table_bytes = sum(n for _, _, n in ranges)
    if args.drop_table and not ranges:
        raise SystemExit(f"no n-gram table tensors (ngram_embedding) in {args.drop_table}'s safetensors files")
    seed = args.seed if args.seed is not None else int(time.time()) % 1_000_000
    report = {"base": base, "model": args.model, "tokens": args.tokens, "tokenizer": tok is not None,
              "table_files": len({p for p, _, _ in ranges}), "table_gib": round(table_bytes / 2**30, 3),
              "runs": []}
    for r in range(args.repeat):
        kinds = ("cold", "warm") if ranges else ("warm",)
        for kind in kinds:
            row: dict = {"round": r, "kind": kind, "seed": seed}
            if ranges and kind == "warm":
                row["rewarm_s"] = round(warm(ranges), 2)
            if ranges and kind == "cold":
                row["drop_s"] = round(drop(ranges), 2)
            row["residency_before"] = residency(ranges) if ranges else None
            content, counted = prompt(text, seed, args.tokens, tok)
            row["tokens_counted"] = counted
            row.update(request(base, args.model, content, args.timeout))
            row["residency_after"] = residency(ranges) if ranges else None
            report["runs"].append(row)
            seed += 1
            print(f"[natural] {kind} round {r}: {row['prompt_tokens']} prompt tokens ({row['cached']} cached), TTFT "
                  f"{row['ttft_s']:.2f} s = {row['tok_s']} tok/s; server prefill {row['server_prefill_s']} s; "
                  + (f"table resident {row['residency_before']:.1%} -> {row['residency_after']:.1%}"
                     if row["residency_before"] is not None else "no table residency"), flush=True)
            args.out.write_text(json.dumps(report, indent=1))

    def med(kind: str, key: str):
        vals = [x[key] for x in report["runs"] if x["kind"] == kind and x.get(key) is not None]
        return statistics.median(vals) if vals else None

    report["summary"] = {"warm_tok_s": med("warm", "tok_s"), "cold_tok_s": med("cold", "tok_s"),
                         "warm_ttft_s": med("warm", "ttft_s"), "cold_ttft_s": med("cold", "ttft_s"),
                         "prompt_tokens": med("warm", "prompt_tokens"),
                         "cold_residency_before": med("cold", "residency_before")}
    args.out.write_text(json.dumps(report, indent=1))
    print(f"[natural] summary: {json.dumps(report['summary'])}", flush=True)


if __name__ == "__main__":
    main()
