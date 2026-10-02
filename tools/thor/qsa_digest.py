"""Digest the time of the QSA kernels inside each prompt range of an Nsight Systems SQLite export.

Reads an Nsight Systems SQLite export: NVTX_EVENTS (prompt ranges -- rows whose text, the text
column or StringIds.value through textId when text is NULL, starts with 'prompt '), StringIds
(short kernel names) and CUPTI_ACTIVITY_KIND_KERNEL (kernel intervals with gridZ). Per prompt
range it reports total kernel time and the split across the QSA kernels.

Usage: python tools/thor/qsa_digest.py DB [--json OUT] [--names a,b,...]
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

QSA_KERNELS = (
    '_chunks', '_merge', '_select', '_select_tiles', '_select_rows', '_scores', '_dequant',
    '_want', '_stage', '_attn_prep', '_attn_gate', '_pool', '_row_attn',
)

_NS_PER_MS = 1e6


def _connect_ro(db):
    path = Path(db).resolve()
    return sqlite3.connect(f'{path.as_uri()}?mode=ro', uri=True)


def _prompt_ranges(conn, strings):
    """NVTX ranges whose text starts with 'prompt ', as (start, end, text) triples."""
    ranges = []
    for start, end, text, text_id in conn.execute('SELECT "start", "end", text, textId FROM NVTX_EVENTS'):
        label = text if text is not None else strings.get(text_id)
        if label is not None and label.startswith('prompt '):
            ranges.append((start, end, label))
    return ranges


def digest(db, names=QSA_KERNELS):
    """Per prompt range: kernel_ms over all enclosed kernels plus n/ms/share for the named ones."""
    wanted = set(names)
    conn = _connect_ro(db)
    try:
        strings = dict(conn.execute('SELECT id, value FROM StringIds'))
        ranges = _prompt_ranges(conn, strings)
        kernels = conn.execute(
            'SELECT shortName, "start", "end", gridZ FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchall()
    finally:
        conn.close()
    out = {}
    for start, end, label in ranges:
        inside = [k for k in kernels if k[1] >= start and k[2] <= end]
        kernel_ms = sum(k[2] - k[1] for k in inside) / _NS_PER_MS
        acc = {}
        for short, kstart, kend, grid_z in inside:
            name = strings.get(short)
            if name not in wanted:
                continue
            key = f'_chunks z{grid_z}' if name == '_chunks' else name
            n, ns = acc.get(key, (0, 0))
            acc[key] = (n + 1, ns + kend - kstart)
        kernels_out = {}
        for key, (n, ns) in acc.items():
            ms = ns / _NS_PER_MS
            kernels_out[key] = {'n': n, 'ms': ms, 'share': ms / kernel_ms if kernel_ms else 0.0}
        qsa_ms = sum(ns for _, ns in acc.values()) / _NS_PER_MS
        out[label] = {'kernel_ms': kernel_ms, 'kernels': kernels_out, 'qsa_ms': qsa_ms,
                      'qsa_share': qsa_ms / kernel_ms if kernel_ms else 0.0}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Per prompt range of an Nsight Systems SQLite export: total kernel time and the '
                    'split across the QSA kernels.')
    parser.add_argument('db', help='path to the Nsight Systems SQLite export')
    parser.add_argument('--json', dest='json_out', metavar='OUT', help='also write the digest as JSON')
    parser.add_argument('--names', default=','.join(QSA_KERNELS), metavar='a,b,...',
                        help='comma separated short names to track (default: all QSA kernels)')
    args = parser.parse_args(argv)
    names = tuple(s for s in (part.strip() for part in args.names.split(',')) if s)
    try:
        d = digest(args.db, names=names)
    except sqlite3.Error as exc:
        print(f'error: cannot read {args.db}: {exc}', file=sys.stderr)
        return 1
    for label, info in d.items():
        print(f"{label}: kernel_ms {info['kernel_ms']:.1f} ms, qsa_ms {info['qsa_ms']:.1f} ms "
              f"({info['qsa_share']:.1%})")
        for key, k in sorted(info['kernels'].items(), key=lambda kv: (-kv[1]['ms'], kv[0])):
            print(f"  {key:26s} n={k['n']:<6} {k['ms']:10.1f} ms  {k['share']:6.1%}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(d, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
