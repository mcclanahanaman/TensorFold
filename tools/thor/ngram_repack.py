"""tools/thor/ngram_repack.py (FR-OPS-17, HD-2026-10-07-04 Phase 2): repacks the NVFP4 n-gram table into one
raw file of fixed records (each row's U8 code bytes, then its F8_E4M3 block-scale bytes; natural row order; no
header, no padding) plus a JSON manifest, and verifies that file record-by-record against the checkpoint.

Commands (python tools/thor/ngram_repack.py <cmd>):
  build MODEL_DIR OUT [--revision R] [--chunk-rows N] [--base B]   prints the manifest's sha256 and rows
  verify MODEL_DIR OUT [--chunk-rows N]                            'OK rows=<n> sha256=<hex>' or 'MISMATCH ...',
                                                                   exiting 1; a bad or missing manifest exits 2

numpy + stdlib only, so it runs on any OS, and memory stays bounded by --chunk-rows: the real table is 320,001,536
rows / 28.8 GB, so no shard or file is ever read whole. Row order and byte offsets are NVMeTable's (thor_nvme.py):
shards ``<base>.shard_<i>.weight`` (or ``<base>.shards.<i>.weight``) in index order, absolute offset =
8 + header length + data_offsets[0].
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FORMAT = 'tensorfold-ngram-rows/1'
DEFAULT_CHUNK = 1 << 20
_SHARD_RES = (re.compile(r'^(?P<base>.*ngram_embedding)\.shard_(?P<idx>\d+)\.weight$'),
              re.compile(r'^(?P<base>.*ngram_embedding)\.shards\.(?P<idx>\d+)\.weight$'))
_SHARD_FIELDS = ('file', 'key', 'rows', 'code_offset', 'scale_offset')


class ManifestError(ValueError):
    """A manifest that does not describe the checkpoint it is verified against."""


def _header(path) -> dict:
    """name -> (dtype, shape, absolute data offset) for one safetensors file (data starts at 8 + header length)."""
    with open(path, 'rb') as f:
        (n,) = struct.unpack('<Q', f.read(8))
        meta = json.loads(f.read(n))
    at = 8 + n
    return {name: (t['dtype'], t['shape'], at + t['data_offsets'][0])
            for name, t in meta.items() if name != '__metadata__'}


def find_tables(model_dir) -> list:
    """The checkpoint's n-gram tables: one {'base', 'shards'} dict per base, sorted by base.

    A shard is a U8 [rows, C] ``<base>.shard_<i>.weight`` (or ``<base>.shards.<i>.weight``) plus its
    ``<same key>.weight_scale`` F8_E4M3 [rows, C // 8], both in the same file; anything else is ignored.
    """
    model_dir = Path(model_dir)
    index = model_dir / 'model.safetensors.index.json'
    if not index.exists():
        return []
    weight_map = json.loads(index.read_text(encoding='utf-8'))['weight_map']
    cache: dict = {}

    def header(file):
        if file not in cache:
            cache[file] = _header(model_dir / file)
        return cache[file]

    found: dict = {}
    for name, file in weight_map.items():
        m = _SHARD_RES[0].match(name) or _SHARD_RES[1].match(name)
        if m is None:
            continue
        h = header(file)
        if name not in h:
            continue
        dtype, shape, code_offset = h[name]
        if dtype != 'U8' or len(shape) != 2:
            continue
        rows, c = shape
        key = name[:-len('.weight')]
        scale = key + '.weight_scale'
        if weight_map.get(scale) != file or scale not in h:
            continue
        sdt, sshape, scale_offset = h[scale]
        if sdt != 'F8_E4M3' or sshape != [rows, c // 8]:
            continue
        shard = {'file': file, 'key': key, 'rows': rows, 'code_offset': code_offset, 'scale_offset': scale_offset,
                 'code_bytes': c, 'scale_bytes': c // 8}
        found.setdefault(m['base'], {})[int(m['idx'])] = shard

    tables = []
    for base in sorted(found):
        shards = found[base]
        n = len(shards)
        if sorted(shards) != list(range(n)):
            raise ValueError(f"{base}: shard indices {sorted(shards)} are not exactly 0..{n - 1}")
        widths = sorted({s['code_bytes'] for s in shards.values()})
        if len(widths) != 1:
            raise ValueError(f"{base}: shards disagree on row width (code_bytes): {widths}")
        tables.append({'base': base, 'shards': [shards[i] for i in range(n)]})
    return tables


def _read_chunk(src, shard, start, n) -> np.ndarray:
    """One chunk of records straight from the shard file: n rows of code bytes then their scale bytes."""
    c, s = shard['code_bytes'], shard['scale_bytes']
    src.seek(shard['code_offset'] + start * c)             # absolute offsets (np.fromfile's offset is relative)
    codes = np.frombuffer(src.read(n * c), dtype=np.uint8)
    src.seek(shard['scale_offset'] + start * s)
    scales = np.frombuffer(src.read(n * s), dtype=np.uint8)
    if codes.size != n * c or scales.size != n * s:
        raise ValueError(f"{shard['file']}: short read at row {start} of {shard['key']}")
    return np.concatenate([codes.reshape(n, c), scales.reshape(n, s)], axis=1)


def build(model_dir, out_path, revision=None, chunk_rows=DEFAULT_CHUNK, base=None) -> dict:
    """Repack MODEL_DIR's n-gram table into OUT (raw records, streamed) and write OUT.json; returns the manifest."""
    model_dir, out_path = Path(model_dir), Path(out_path)
    tables = find_tables(model_dir)
    if not tables:
        raise ValueError(f'no NVFP4 n-gram table in {model_dir}')
    if base is None:
        if len(tables) > 1:
            raise ValueError('several NVFP4 n-gram tables in {}: {} (pass base)'.format(
                model_dir, ', '.join(t['base'] for t in tables)))
        table = tables[0]
    else:
        table = next((t for t in tables if t['base'] == base), None)
        if table is None:
            raise ValueError(f'no NVFP4 n-gram table named {base!r} in {model_dir}')
    shards = table['shards']
    c, s = shards[0]['code_bytes'], shards[0]['scale_bytes']
    record = c + s
    if revision is None:
        revision = model_dir.name if model_dir.parent.name == 'snapshots' else None

    h = hashlib.sha256()
    tmp = out_path.parent / (out_path.name + '.tmp')
    with open(tmp, 'wb') as out:
        for sh in shards:
            with open(model_dir / sh['file'], 'rb') as src:
                for start in range(0, sh['rows'], chunk_rows):
                    n = min(chunk_rows, sh['rows'] - start)
                    blob = _read_chunk(src, sh, start, n).tobytes()
                    out.write(blob)
                    h.update(blob)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, out_path)

    manifest = {
        'format': FORMAT,
        'base': table['base'],
        'rows': sum(sh['rows'] for sh in shards),
        'record_bytes': record,
        'code_bytes': c,
        'scale_bytes': s,
        'size': out_path.stat().st_size,
        'sha256': h.hexdigest(),
        'revision': revision,
        'shards': [dict({k: sh[k] for k in _SHARD_FIELDS}, file_size=(model_dir / sh['file']).stat().st_size)
                   for sh in shards],
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    with open(str(out_path) + '.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=1)
        f.write('\n')
    return manifest


def verify(model_dir, out_path, chunk_rows=DEFAULT_CHUNK) -> dict:
    """Compare OUT and its manifest with the checkpoint: {'ok': True, 'rows', 'sha256'} or a failure dict."""
    model_dir, out_path = Path(model_dir), Path(out_path)
    manifest_path = Path(str(out_path) + '.json')
    if not manifest_path.exists():
        raise ManifestError(f'no manifest {manifest_path} for {out_path}')
    m = json.loads(manifest_path.read_text(encoding='utf-8'))

    table = next((t for t in find_tables(model_dir) if t['base'] == m.get('base')), None)
    if table is None:
        raise ManifestError(f"manifest names base {m.get('base')!r}, which {model_dir} does not contain")
    shards = table['shards']
    c, s = shards[0]['code_bytes'], shards[0]['scale_bytes']
    bad = []
    if m.get('rows') != sum(sh['rows'] for sh in shards):
        bad.append(f"rows (manifest {m.get('rows')!r} != checkpoint {sum(sh['rows'] for sh in shards)})")
    if m.get('record_bytes') != c + s:
        bad.append(f"record_bytes (manifest {m.get('record_bytes')!r} != checkpoint {c + s})")
    if m.get('code_bytes') != c:
        bad.append(f"code_bytes (manifest {m.get('code_bytes')!r} != checkpoint {c})")
    if m.get('scale_bytes') != s:
        bad.append(f"scale_bytes (manifest {m.get('scale_bytes')!r} != checkpoint {s})")
    msh = m.get('shards')
    if not isinstance(msh, list) or len(msh) != len(shards):
        bad.append(f'shards (manifest {msh!r} vs {len(shards)} checkpoint shards)')
    else:
        for j, (sh, ms) in enumerate(zip(shards, msh)):
            want = dict({k: sh[k] for k in _SHARD_FIELDS}, file_size=(model_dir / sh['file']).stat().st_size)
            for k, v in want.items():
                got = ms.get(k) if isinstance(ms, dict) else None
                if got != v:
                    bad.append(f'shard {j} {k} (manifest {got!r} != checkpoint {v!r})')
    if bad:
        raise ManifestError('manifest does not describe the checkpoint: ' + '; '.join(bad))

    rows, record = m['rows'], m['record_bytes']
    size = out_path.stat().st_size
    if size != rows * record:
        return {'ok': False, 'reason': 'size', 'size': size, 'expected': rows * record, 'rows': rows}
    h = hashlib.sha256()
    row = 0
    with open(out_path, 'rb') as got_f:
        for sh in shards:
            with open(model_dir / sh['file'], 'rb') as src:
                for start in range(0, sh['rows'], chunk_rows):
                    n = min(chunk_rows, sh['rows'] - start)
                    want = _read_chunk(src, sh, start, n)
                    blob = got_f.read(n * record)
                    got = np.frombuffer(blob, dtype=np.uint8)
                    if got.size != n * record:
                        return {'ok': False, 'reason': 'row', 'first_bad_row': row, 'rows': rows}
                    got = got.reshape(n, record)
                    if not np.array_equal(got, want):
                        first = row + int(np.argmax(np.any(got != want, axis=1)))
                        return {'ok': False, 'reason': 'row', 'first_bad_row': first, 'rows': rows}
                    h.update(blob)
                    row += n
    sha = h.hexdigest()
    return {'ok': sha == m.get('sha256'), 'rows': rows, 'sha256': sha}


def _cmd_build(args) -> int:
    m = build(args.model_dir, args.out, revision=args.revision, chunk_rows=args.chunk_rows, base=args.base)
    print(f"sha256={m['sha256']} rows={m['rows']}")
    return 0


def _cmd_verify(args) -> int:
    try:
        r = verify(args.model_dir, args.out, chunk_rows=args.chunk_rows)
    except ManifestError as e:
        print(e, file=sys.stderr)
        return 2
    if not r['ok']:
        if 'first_bad_row' in r:
            print(f"MISMATCH row {r['first_bad_row']}")
        elif r.get('reason') == 'size':
            print('MISMATCH size')
        else:
            print('MISMATCH sha256')
        return 1
    print(f"OK rows={r['rows']} sha256={r['sha256']}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='repack and verify the NVFP4 n-gram table rows (FR-OPS-17)')
    sub = p.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build', help='repack the n-gram table into OUT and write OUT.json')
    b.add_argument('model_dir')
    b.add_argument('out')
    b.add_argument('--revision', default=None, help='revision id to record (default: the snapshot dir name)')
    b.add_argument('--chunk-rows', type=int, default=DEFAULT_CHUNK, help='rows read per chunk (default %(default)s)')
    b.add_argument('--base', default=None, help='table base, when the checkpoint has several')
    b.set_defaults(func=_cmd_build)
    v = sub.add_parser('verify', help='verify OUT record-by-record against the checkpoint')
    v.add_argument('model_dir')
    v.add_argument('out')
    v.add_argument('--chunk-rows', type=int, default=DEFAULT_CHUNK, help='rows read per chunk (default %(default)s)')
    v.set_defaults(func=_cmd_verify)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
