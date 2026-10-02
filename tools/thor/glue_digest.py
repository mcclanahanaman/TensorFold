"""Digest two kernel dumps into the MoE glue's kernel classes per prompt, OFF versus ON.

Reads two kernel dumps -- JSON objects mapping a prompt label to {'kernel_ms', 'wall_ms',
'kernels'} -- and reports, per prompt, the launch count and milliseconds of every kernel class of
the MoE glue's per-token pipeline -- router, topk, plan, tiles, gather, the expert GEMM, scatter,
the shared swiglu, the writeback and the torch elementwise remainder -- OFF versus ON, with the
glue stages' total, its share of kernel time and the deltas. A kernel's class comes from its
short name (the part of its 'key' before the first '|'); anything unrecognized is 'other'.

Usage: python tools/thor/glue_digest.py OFF.json ON.json [--json FILE] [--markdown FILE]
"""

import argparse
import json
import sys
from pathlib import Path

CLASSES = ('router', 'topk', 'plan', 'tiles', 'gather', 'experts_gemm', 'scatter', 'shared_swiglu',
           'writeback', 'torch_elementwise', 'other')
GLUE = ('router', 'topk', 'plan', 'tiles', 'gather', 'scatter', 'shared_swiglu')


def classify(name: str) -> str:
    """Map a kernel's short name to its glue class; anything unrecognized is 'other'."""
    if '_router' in name:
        return 'router'
    if '_topk_rows' in name:
        return 'topk'
    if name.startswith('plan_'):
        return 'plan'
    if 'tiles' in name:
        return 'tiles'
    if 'gather' in name:
        return 'gather'
    if 'scatter' in name:
        return 'scatter'
    if 'W4A16GroupedGemmKernel' in name:
        return 'experts_gemm'
    if '_swiglu' in name:
        return 'shared_swiglu'
    if '_hc_writeback' in name or '_writeback_rows' in name:
        return 'writeback'
    if 'elementwise_kernel' in name:  # also matches the vectorized_ and unrolled_ variants
        return 'torch_elementwise'
    return 'other'


def _tally(entry: dict) -> dict:
    """Sum n and ms per class over one dump entry's kernels, every class of CLASSES present."""
    acc = {cls: [0, 0.0] for cls in CLASSES}
    for kernel in entry['kernels']:
        cls = classify(kernel['key'].split('|', 1)[0])
        acc[cls][0] += kernel['n']
        acc[cls][1] += kernel['ms']
    return acc


def digest(off: dict, on: dict) -> dict:
    """Per prompt label in both dumps: class table, totals and glue shares, OFF versus ON.

    'unmatched' lists, sorted, the prompt labels present in only one of the two dumps.
    """
    unmatched = sorted(set(off) ^ set(on))
    prompts = {}
    for label in off:
        if label not in on:
            continue
        off_tally, on_tally = _tally(off[label]), _tally(on[label])
        classes = {cls: {'n_off': off_tally[cls][0], 'ms_off': off_tally[cls][1],
                         'n_on': on_tally[cls][0], 'ms_on': on_tally[cls][1],
                         'delta_ms': on_tally[cls][1] - off_tally[cls][1]} for cls in CLASSES}
        kernel_ms_off, kernel_ms_on = off[label]['kernel_ms'], on[label]['kernel_ms']
        glue_ms_off = sum(classes[cls]['ms_off'] for cls in GLUE)
        glue_ms_on = sum(classes[cls]['ms_on'] for cls in GLUE)
        prompts[label] = {
            'classes': classes,
            'kernel_ms_off': kernel_ms_off, 'kernel_ms_on': kernel_ms_on,
            'kernel_ms_delta': kernel_ms_on - kernel_ms_off,
            'wall_ms_off': off[label]['wall_ms'], 'wall_ms_on': on[label]['wall_ms'],
            'glue_ms_off': glue_ms_off, 'glue_ms_on': glue_ms_on,
            'glue_share_off': 100.0 * glue_ms_off / kernel_ms_off if kernel_ms_off else 0.0,
            'glue_share_on': 100.0 * glue_ms_on / kernel_ms_on if kernel_ms_on else 0.0,
        }
    return {'prompts': prompts, 'unmatched': unmatched}


def _totals_row(name: str, ms_off: float, ms_on: float) -> str:
    """A totals row: both n cells empty, the ms cells and their delta filled."""
    return f'| {name} | | {ms_off:.1f} | | {ms_on:.1f} | {ms_on - ms_off:.1f} |'


def markdown(result: dict) -> str:
    """Render the digest as one markdown table per prompt, plus an unmatched-labels line."""
    lines = []
    for label, entry in result['prompts'].items():
        lines += [f'## {label}',
                  '| class | n off | ms off | n on | ms on | delta ms |',
                  '|---|---|---|---|---|---|']
        for cls in CLASSES:
            c = entry['classes'][cls]
            cells = (str(c['n_off']), f"{c['ms_off']:.1f}", str(c['n_on']),
                     f"{c['ms_on']:.1f}", f"{c['delta_ms']:.1f}")
            lines.append(f'| {cls} | ' + ' | '.join(cells) + ' |')
        lines.append(_totals_row('glue', entry['glue_ms_off'], entry['glue_ms_on']))
        lines.append(_totals_row('kernels', entry['kernel_ms_off'], entry['kernel_ms_on']))
    if result['unmatched']:
        lines.append('unmatched: ' + ', '.join(result['unmatched']))
    return '\n'.join(lines) + '\n'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Side-by-side digest of two kernel dumps: the MoE glue's kernel classes per "
                    'prompt, OFF versus ON.')
    parser.add_argument('off', help='path to the OFF dump JSON')
    parser.add_argument('on', help='path to the ON dump JSON')
    parser.add_argument('--json', dest='json_out', metavar='FILE',
                        help='also write the digest as JSON')
    parser.add_argument('--markdown', dest='markdown_out', metavar='FILE',
                        help='also write the digest as markdown')
    args = parser.parse_args(argv)
    off = json.loads(Path(args.off).read_text(encoding='utf-8'))
    on = json.loads(Path(args.on).read_text(encoding='utf-8'))
    result = digest(off, on)
    text = markdown(result)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    if args.markdown_out:
        Path(args.markdown_out).write_text(text, encoding='utf-8')
    print(text, end='')
    return 0


if __name__ == '__main__':
    sys.exit(main())
