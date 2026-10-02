"""Digest two kernel dumps into a side-by-side view of the hyper-connection kernels per prompt.

Reads two kernel dumps -- JSON objects mapping a prompt label to {'kernel_ms', 'wall_ms',
'kernels'} -- and reports, per prompt, the launch count, milliseconds and share of kernel time of
every hyper-connection op, OFF versus ON, with the family total and the milliseconds saved. Every
_b16mm and _reduce launch counts as the hyper-connection down projection (on an NVFP4 checkpoint's
prompt passes it is their only user).

Usage: python tools/thor/hc_digest.py OFF.json ON.json [--json FILE] [--markdown FILE]
"""

import argparse
import json
import sys
from pathlib import Path

OPS = (
    ('_hc_writeback', 'writeback'),
    ('_hc_normed', 'normed'),
    ('_b16mm', 'down (b16 split-K)'),
    ('_reduce', 'down reduce'),
    ('_hc_act', 'act'),
    ('_hc_reduce_act', 'reduce + act'),
    ('_hc_mix', 'mix'),
    ('_down_act', 'down + act (fused)'),
    ('_mix_h', 'mix from h'),
    ('_upmix_h', 'up + mix (fused)'),
    ('_upmix4', 'up + mix (fused)'),
)

_OPS_BY_NAME = dict(OPS)
_LABELS = tuple(dict.fromkeys(label for _, label in OPS))      # op labels in order, each once


def ops(entry: dict) -> dict:
    """Sum n and ms per op label over one entry's kernels; grids counts the distinct keys."""
    acc = {label: [0, 0.0, set()] for label in _LABELS}
    for kernel in entry['kernels']:
        label = _OPS_BY_NAME.get(kernel['key'].split('|', 1)[0])
        if label is not None:
            n, ms, keys = acc[label]
            acc[label] = [n + kernel['n'], ms + kernel['ms'], keys | {kernel['key']}]
    return {label: {'n': n, 'ms': ms, 'grids': len(keys)}
            for label, (n, ms, keys) in acc.items() if keys}


def _side(entry):
    """One dump's side of a digest row: family total, its share of kernel time and the op split."""
    if entry is None:
        return None
    family = ops(entry)
    family_ms = sum((op['ms'] for op in family.values()), 0.0)
    kernel_ms = entry['kernel_ms']
    return {'kernel_ms': kernel_ms, 'family_ms': family_ms,
            'family_n': sum(op['n'] for op in family.values()),
            'family_share': family_ms / kernel_ms if kernel_ms else 0.0, 'ops': family}


def digest(off: dict, on: dict) -> dict:
    """Per prompt label: both sides with the family total and the milliseconds saved OFF -> ON."""
    d = {}
    labels = list(off) + [label for label in on if label not in off]
    for label in labels:
        off_side, on_side = _side(off.get(label)), _side(on.get(label))
        saved_ms = kernel_saved_ms = None
        if off_side is not None and on_side is not None:
            saved_ms = off_side['family_ms'] - on_side['family_ms']
            kernel_saved_ms = off_side['kernel_ms'] - on_side['kernel_ms']
        d[label] = {'off': off_side, 'on': on_side, 'saved_ms': saved_ms,
                    'kernel_saved_ms': kernel_saved_ms}
    return d


def _cells(side, op=None):
    """Three cells for one side: the family row when op is None, else that op's row."""
    if side is None:
        return ['-', '-', '-']
    if op is None:
        return [str(side['family_n']), f"{side['family_ms']:.1f}", f"{side['family_share']:.1%}"]
    kernel = side['ops'].get(op)
    if kernel is None:
        return ['0', '0.0', '0.0%']
    share = kernel['ms'] / side['kernel_ms'] if side['kernel_ms'] else 0.0
    return [str(kernel['n']), f"{kernel['ms']:.1f}", f"{share:.1%}"]


def markdown(d: dict) -> str:
    """Render the digest as a markdown table: one row per op with launches plus the family row."""
    lines = ['| prompt | op | n off | ms off | share off | n on | ms on | share on |',
             '|---|---|---|---|---|---|---|---|']
    for label, sides in d.items():
        off, on = sides['off'], sides['on']
        for op in _LABELS:
            if (off is not None and op in off['ops']) or (on is not None and op in on['ops']):
                lines.append(f'| {label} | {op} | '
                             + ' | '.join(_cells(off, op) + _cells(on, op)) + ' |')
        lines.append(f'| {label} | hc family | ' + ' | '.join(_cells(off) + _cells(on)) + ' |')
    return '\n'.join(lines) + '\n'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Side-by-side digest of two kernel dumps: the hyper-connection kernels per '
                    'prompt with the family total and its share of kernel time.')
    parser.add_argument('off', help='path to the OFF dump JSON')
    parser.add_argument('on', help='path to the ON dump JSON')
    parser.add_argument('--json', dest='json_out', metavar='FILE',
                        help='also write the digest as JSON')
    parser.add_argument('--markdown', dest='markdown_out', metavar='FILE',
                        help='also write the digest as a markdown table')
    args = parser.parse_args(argv)
    off = json.loads(Path(args.off).read_text(encoding='utf-8'))
    on = json.loads(Path(args.on).read_text(encoding='utf-8'))
    d = digest(off, on)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(d, indent=1) + '\n', encoding='utf-8')
    text = markdown(d)
    if args.markdown_out:
        Path(args.markdown_out).write_text(text, encoding='utf-8')
    print(text, end='')
    return 0


if __name__ == '__main__':
    sys.exit(main())
