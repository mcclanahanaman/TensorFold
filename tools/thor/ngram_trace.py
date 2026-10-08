"""tools/thor/ngram_trace.py (FR-OPS-17, HD-2026-10-07-04 Phase 0): turn chat/agent traffic into the exact Flash
Next PLE n-gram row ids (TensorFold's own hash) and replay them through a row-cache policy for hit rates.

Commands (python tools/thor/ngram_trace.py <cmd>):
  extract --source dsh|jsonl --input PATH --tokenizer TOK.json --out DIR [--target-tokens N] [--seed S]
  simulate DIR --streams 8 --chunk 256 --sizes 4,6,8,16 --probation 0,2,4 --policy nvme|lru --out FILE

numpy is the only import-time dependency; the tokenizer and the NVMe policy import lazily inside the code that
needs them. The n-gram module is loaded by file path (never via the tensorfold package).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

NGRAM_ARGS = dict(vocab=248320, ngram_size=3, heads_per_ngram=8, vocab_base=20_000_000, divisor=128, shards=128,
                  seed=1234, eos=248044, embed_dim=2560, ple_index=0)

_DSH_TYPES = {'system/message', 'assistant/message', 'tool/call', 'tool/result',
              'agent/inbox/spliced', 'user/message'}
_DSH_ROLES = {'assistant/message': 'assistant', 'tool/call': 'assistant', 'tool/result': 'tool',
              'system/message': 'system'}
_DSH_SKIP_KEYS = {'replayState', 'stream', 'usage', 'meta', 'source'}
_DSH_KEEP_KEYS = ('text', 'arguments')


def make_ngram():
    path = Path(__file__).resolve().parents[2] / 'src/tensorfold/families/qwen4_exp/cuda/ngram.py'
    spec = importlib.util.spec_from_file_location('tensorfold_qwen4_ngram_host', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NGram(**NGRAM_ARGS)


def session_rows(messages, encode, ngram):
    ids, src = [], []
    for role, text in messages:
        chunk = encode(f'<|im_start|>{role}\n{text}<|im_end|>\n')
        ids += chunk
        src += [1 if role == 'assistant' else 0] * len(chunk)
    rows = ngram.ids(ngram.initial_history(), np.asarray(ids, dtype=np.int64)).astype(np.uint32)
    return rows, np.asarray(src, dtype=np.uint8)


def _dsh_strings(node, out):
    if isinstance(node, str):
        out.append(node)
    elif isinstance(node, list):
        for item in node:
            _dsh_strings(item, out)
    elif isinstance(node, dict):
        for key, item in node.items():
            if key not in _DSH_SKIP_KEYS:
                _dsh_strings(item, out)


def _dsh_walk(node, out):
    if isinstance(node, dict):
        for key, item in node.items():
            if key in _DSH_SKIP_KEYS:
                continue
            if key in _DSH_KEEP_KEYS:
                _dsh_strings(item, out)
            else:
                _dsh_walk(item, out)
    elif isinstance(node, list):
        for item in node:
            _dsh_walk(item, out)


def dsh_messages(path):
    out = []
    with open(path, encoding='utf-8', errors='replace') as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get('type') not in _DSH_TYPES:
                continue
            strings = []
            _dsh_walk(event.get('data'), strings)
            if not strings:
                continue
            out.append((_DSH_ROLES.get(event['type'], 'user'), '\n'.join(strings)))
    return out


def jsonl_messages(obj):
    if 'messages' in obj:
        return [(m['role'], m['content']) for m in obj['messages']]
    if 'prompt' in obj:
        messages = [('user', obj['prompt'])]
        if obj.get('response'):
            messages.append(('assistant', obj['response']))
        return messages
    return [('user', obj['text'])]


def write_trace(out_dir, sessions):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    row_blocks, src_blocks, meta = [], [], []
    for name, rows, src in sessions:
        rows = np.asarray(rows, dtype=np.uint32)
        src = np.asarray(src, dtype=np.uint8)
        row_blocks.append(rows.reshape(-1))
        src_blocks.append(src)
        meta.append({'name': name, 'tokens': int(rows.shape[0]), 'assistant_tokens': int(src.sum())})
    rows_all = np.concatenate(row_blocks) if row_blocks else np.zeros(0, dtype=np.uint32)
    src_all = np.concatenate(src_blocks) if src_blocks else np.zeros(0, dtype=np.uint8)
    rows_all.tofile(out / 'rows.u32')
    src_all.tofile(out / 'src.u8')
    payload = {'heads': 16, 'sessions': meta}
    (out / 'sessions.json').write_text(json.dumps(payload))
    return payload


def read_trace(dir):
    dir = Path(dir)
    rows = np.fromfile(dir / 'rows.u32', dtype=np.uint32).reshape(-1, 16)
    src = np.fromfile(dir / 'src.u8', dtype=np.uint8)
    sessions = json.loads((dir / 'sessions.json').read_text())['sessions']
    return rows, src, sessions


def interleave(lengths, streams, chunk):
    """Round-robin replay order over back-to-back sessions: a free stream takes the next unstarted session at the
    start of its turn and then emits min(chunk, remaining) global token indices per turn of its own."""
    lengths = [int(x) for x in lengths]
    total = sum(lengths)
    offsets = [0] * len(lengths)
    for i in range(1, len(lengths)):
        offsets[i] = offsets[i - 1] + lengths[i - 1]
    order = np.empty(total, dtype=np.int64)
    filled, next_session, turn = 0, 0, 0
    active = [None] * streams                       # per stream: [start, stop) cursor into global indices
    while filled < total:
        j = turn % streams
        turn += 1
        cursor = active[j]
        if cursor is None:
            if next_session >= len(lengths):
                continue
            k = next_session
            next_session += 1
            cursor = [offsets[k], offsets[k] + lengths[k]]
            active[j] = cursor
        take = min(chunk, cursor[1] - cursor[0])
        if take > 0:
            order[filled:filled + take] = np.arange(cursor[0], cursor[0] + take, dtype=np.int64)
            filled += take
            cursor[0] += take
        if cursor[0] == cursor[1]:
            active[j] = None                        # session done; next one is picked up next turn
    return order


def nvme_policy(cache_rows, probation, init):
    if sys.platform != 'linux':
        raise RuntimeError('nvme policy needs Linux (src/tensorfold/families/qwen4_exp/cuda/thor_nvme.py)')
    import tensorfold.families.qwen4_exp.cuda.thor_nvme as thor_nvme
    return thor_nvme.NVMeTable.policy(cache_rows, probation, init)


class _LRU:
    """Exact LRU over row ids; simulate(ids) -> uint8 hits (probation and init ignored)."""

    def __init__(self, rows: int) -> None:
        self.rows = rows
        self.d = OrderedDict()

    def simulate(self, ids):
        hit = np.zeros(len(ids), dtype=np.uint8)
        for i, x in enumerate(ids.tolist()):
            if x in self.d:
                self.d.move_to_end(x)
                hit[i] = 1
            else:
                self.d[x] = None
                if len(self.d) > self.rows:
                    self.d.popitem(last=False)
        return hit


def lru_policy(cache_rows, probation, init):
    return _LRU(cache_rows)


def simulate(rows, src, order, cache_rows, probation, init=0, policy=None):
    table = (policy or nvme_policy)(cache_rows, probation, init)
    order = np.asarray(order, dtype=np.int64)
    ids = np.asarray(rows, dtype=np.uint32)[order].reshape(-1).astype(np.int64)
    step = 1 << 22
    chunks = [table.simulate(ids[i:i + step]) for i in range(0, ids.size, step)]
    hits = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint8)
    h = hits.reshape(-1, 16)
    s = np.asarray(src, dtype=np.uint8)[order]
    generated, prompted = s == 1, s == 0
    return {
        'lookups': int(ids.size),
        'hit_all': float(h.mean()) if h.size else 0.0,
        'hit_prompt': float(h[prompted].mean()) if prompted.any() else 0.0,
        'hit_generated': float(h[generated].mean()) if generated.any() else 0.0,
        'hit_bigram': float(h[:, :8].mean()) if h.size else 0.0,
        'hit_trigram': float(h[:, 8:].mean()) if h.size else 0.0,
        'p_gen_token_miss': float((h[generated] != 1).any(axis=1).mean()) if generated.any() else 0.0,
    }


def piece_unique(rows, lengths, piece):
    counts, start = [], 0
    for length in lengths:
        length = int(length)
        for begin in range(start, start + length, piece):
            counts.append(np.unique(rows[begin:min(begin + piece, start + length)]).size)
        start += length
    arr = np.asarray(counts, dtype=np.float64)
    return {'pieces': len(counts), 'mean': float(arr.mean()), 'p50': float(np.percentile(arr, 50)),
            'p90': float(np.percentile(arr, 90)), 'max': int(arr.max())}


def _parse_list(text, cast):
    return [cast(part) for part in text.split(',') if part.strip()]


def cmd_extract(args):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    encode = lambda s: tok.encode(s, add_special_tokens=False).ids  # noqa: E731
    ngram = make_ngram()
    sessions, total = [], 0

    def accept(name, messages):
        nonlocal total
        rows, src = session_rows(messages, encode, ngram)
        if len(src) < 64:
            return
        sessions.append((name, rows, src))
        total += int(len(src))

    if args.source == 'dsh':
        jobs = sorted(p for p in Path(args.input).iterdir() if p.is_dir())
        random.Random(args.seed).shuffle(jobs)
        for job in jobs:
            files = sorted(job.glob('*.native.jsonl'))
            if files:
                accept(job.name, dsh_messages(files[0]))
                if total >= args.target_tokens:
                    break
    else:
        path = Path(args.input)
        files = [path] if path.is_file() else sorted(path.glob('*.jsonl'))
        for file in files:
            with open(file, encoding='utf-8', errors='replace') as handle:
                for number, line in enumerate(handle, 1):
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    accept(f'{file.name}:{number}', jsonl_messages(obj))
                    if total >= args.target_tokens:
                        break
            if total >= args.target_tokens:
                break
    summary = write_trace(args.out, sessions)
    tokens = sum(int(s['tokens']) for s in summary['sessions'])
    assistant = sum(int(s['assistant_tokens']) for s in summary['sessions'])
    print(f"{args.source}: {len(summary['sessions'])} sessions, {tokens} tokens, "
          f"{assistant} assistant tokens -> {args.out}")


def cmd_simulate(args):
    rows, src, sessions = read_trace(args.dir)
    lengths = [int(s['tokens']) for s in sessions]
    order = interleave(lengths, args.streams, args.chunk)
    sizes = _parse_list(args.sizes, float)
    probations = _parse_list(args.probation, int)
    policy = lru_policy if args.policy == 'lru' else None
    cells = []
    for probation in probations:
        for size in sizes:
            cache_rows = int(round(size * 1e6))
            cell = simulate(rows, src, order, cache_rows, probation, policy=policy)
            cell['cache_rows'] = cache_rows
            cell['probation'] = probation
            cell['cache_gib'] = cache_rows * 95 / 2 ** 30
            cells.append(cell)
            print(f"probation={probation} cache_rows={cache_rows} cache_gib={cell['cache_gib']:.2f} "
                  f"hit_all={cell['hit_all']:.4f} hit_prompt={cell['hit_prompt']:.4f} "
                  f"hit_generated={cell['hit_generated']:.4f} hit_bigram={cell['hit_bigram']:.4f} "
                  f"hit_trigram={cell['hit_trigram']:.4f} p_gen_token_miss={cell['p_gen_token_miss']:.4f}")
    payload = {'tokens': int(rows.shape[0]), 'streams': args.streams, 'chunk': args.chunk, 'policy': args.policy,
               'cells': cells, 'piece4096': piece_unique(rows, lengths, 4096),
               'piece2048': piece_unique(rows, lengths, 2048)}
    Path(args.out).write_text(json.dumps(payload, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description='PLE n-gram trace extraction and cache-policy replay (FR-OPS-17)')
    sub = parser.add_subparsers(dest='cmd', required=True)
    extract = sub.add_parser('extract', help='traffic -> PLE n-gram row ids on disk')
    extract.add_argument('--source', choices=['dsh', 'jsonl'], required=True)
    extract.add_argument('--input', required=True)
    extract.add_argument('--tokenizer', required=True)
    extract.add_argument('--out', required=True)
    extract.add_argument('--target-tokens', type=int, default=5_000_000)
    extract.add_argument('--seed', type=int, default=20261007)
    extract.set_defaults(func=cmd_extract)
    sim = sub.add_parser('simulate', help='replay a trace directory through a row-cache policy')
    sim.add_argument('dir')
    sim.add_argument('--streams', type=int, default=8)
    sim.add_argument('--chunk', type=int, default=256)
    sim.add_argument('--sizes', default='4,6,8,16')
    sim.add_argument('--probation', default='0,2,4')
    sim.add_argument('--policy', choices=['nvme', 'lru'], default='nvme')
    sim.add_argument('--out', required=True)
    sim.set_defaults(func=cmd_simulate)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
