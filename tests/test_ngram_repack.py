"""tools/thor/ngram_repack.py: the NVFP4 n-gram rows repacked into one raw file of
fixed records (a row's code bytes, then its block-scale bytes; natural row order; no header, no padding) plus a JSON
manifest, and a verifier that compares every record with the checkpoint. The contract the tool is written to (GLM card
tk-ngram-repack); numpy + stdlib only, so it runs on any OS. Row order and byte offsets are NVMeTable's (thor_nvme.py):
shards ``<base>.shard_<i>`` (or ``<base>.shards.<i>``) in index order, offset = 8 + header length + data_offsets[0]."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "thor" / "ngram_repack.py"
spec = importlib.util.spec_from_file_location("ngram_repack", TOOL)
nr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nr)

BASE = "model.language_model.layers.2.ple_embedding.ngram_embedding"
C = 80                                   # code bytes a row (160 values); scales C // 8 = 10 bytes


def write_st(path: Path, tensors: list[tuple[str, str, list[int], bytes]]) -> None:
    """A safetensors file from (name, dtype, shape, raw bytes), in the given data order."""

    header, at = {}, 0
    for name, dtype, shape, raw in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [at, at + len(raw)]}
        at += len(raw)
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob)
        for *_, raw in tensors:
            f.write(raw)


def make_ckpt(tmp: Path, rows=(7, 5, 9, 4, 6), spelling="shard_{}", width=C, seed=0, drop=None, narrow=None):
    """A two-file checkpoint: the shards split across files out of index order, unrelated tensors between them."""

    rng = np.random.default_rng(seed)
    snap = tmp / "snapshots" / "rev0abc"
    snap.mkdir(parents=True)
    wid = [width - 16 if i == narrow else width for i in range(len(rows))]
    codes = [rng.integers(0, 256, (r, w), dtype=np.uint8) for r, w in zip(rows, wid)]
    scales = [rng.integers(0, 256, (r, w // 8), dtype=np.uint8) for r, w in zip(rows, wid)]
    files = {"model-00001-of-00002.safetensors": [], "model-00002-of-00002.safetensors": []}
    order = [3, 0, 4, 1, 2][:len(rows)] if len(rows) == 5 else list(range(len(rows)))[::-1]
    wm = {}
    for n, i in enumerate(order):
        if drop is not None and i == drop:
            continue
        f = "model-00001-of-00002.safetensors" if n % 2 == 0 else "model-00002-of-00002.safetensors"
        key = f"{BASE}.{spelling.format(i)}"
        files[f].append((f"other.{i}.weight", "BF16", [3, 4], rng.integers(0, 256, 24, dtype=np.uint8).tobytes()))
        files[f].append((key + ".weight_scale", "F8_E4M3", [rows[i], wid[i] // 8], scales[i].tobytes()))
        files[f].append((key + ".weight", "U8", [rows[i], wid[i]], codes[i].tobytes()))
        for t in files[f][-3:]:
            wm[t[0]] = f
    for f, ts in files.items():
        write_st(snap / f, ts)
    (snap / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    want = np.concatenate([np.concatenate([c, s], axis=1) for c, s in zip(codes, scales)]) if narrow is None else None
    return snap, want


def cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOL), *map(str, args)], capture_output=True, text=True)


def test_find_tables_order_and_offsets(tmp_path):
    snap, want = make_ckpt(tmp_path)
    tables = nr.find_tables(snap)
    assert [t["base"] for t in tables] == [BASE]
    sh = tables[0]["shards"]
    assert [s["key"] for s in sh] == [f"{BASE}.shard_{i}" for i in range(5)]
    assert [s["rows"] for s in sh] == [7, 5, 9, 4, 6]
    assert all(s["code_bytes"] == C and s["scale_bytes"] == C // 8 for s in sh)
    at = 0
    for s in sh:                                   # the offsets address the bytes exactly (NVMeTable's arithmetic)
        raw = (snap / s["file"]).read_bytes()
        got = np.frombuffer(raw[s["code_offset"]:s["code_offset"] + s["rows"] * C], np.uint8).reshape(-1, C)
        sc = np.frombuffer(raw[s["scale_offset"]:s["scale_offset"] + s["rows"] * C // 8], np.uint8).reshape(-1, C // 8)
        assert np.array_equal(np.concatenate([got, sc], axis=1), want[at:at + s["rows"]])
        at += s["rows"]


def test_build_records_and_manifest(tmp_path):
    snap, want = make_ckpt(tmp_path)
    out = tmp_path / "rows" / "ngram.rows"
    out.parent.mkdir()
    m = nr.build(snap, out)
    data = out.read_bytes()
    assert len(data) == want.size == 31 * (C + C // 8)
    assert np.array_equal(np.frombuffer(data, np.uint8).reshape(-1, C + C // 8), want)
    disk = json.loads(Path(str(out) + ".json").read_text())
    assert disk == m
    assert m["format"] == "tensorfold-ngram-rows/1" and m["base"] == BASE and m["rows"] == 31
    assert (m["record_bytes"], m["code_bytes"], m["scale_bytes"]) == (C + C // 8, C, C // 8)
    assert m["size"] == len(data) and m["sha256"] == hashlib.sha256(data).hexdigest()
    assert m["revision"] == "rev0abc"                       # the snapshot directory's name unless --revision
    assert [s["rows"] for s in m["shards"]] == [7, 5, 9, 4, 6]
    assert all(s["file_size"] == (snap / s["file"]).stat().st_size for s in m["shards"])
    assert not list(out.parent.glob("*.tmp"))               # written through a temporary file, then renamed


def test_chunking_does_not_change_the_file(tmp_path):
    snap, _ = make_ckpt(tmp_path)
    a, b = tmp_path / "a.rows", tmp_path / "b.rows"
    assert nr.build(snap, a, chunk_rows=3)["sha256"] == nr.build(snap, b, chunk_rows=1 << 20)["sha256"]
    assert a.read_bytes() == b.read_bytes()


def test_nested_spelling_and_revision_arg(tmp_path):
    snap, want = make_ckpt(tmp_path, spelling="shards.{}")
    out = tmp_path / "n.rows"
    r = cli("build", snap, out, "--revision", "7c4f1bc1")
    assert r.returncode == 0, r.stderr
    assert np.array_equal(np.fromfile(out, np.uint8).reshape(-1, C + C // 8), want)
    assert json.loads(Path(str(out) + ".json").read_text())["revision"] == "7c4f1bc1"


def test_verify_passes_then_names_the_first_bad_row(tmp_path):
    snap, _ = make_ckpt(tmp_path)
    out = tmp_path / "v.rows"
    assert cli("build", snap, out).returncode == 0
    ok = cli("verify", snap, out, "--chunk-rows", "4")
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "OK" in ok.stdout and json.loads(Path(str(out) + ".json").read_text())["sha256"] in ok.stdout
    data = bytearray(out.read_bytes())
    data[13 * (C + C // 8) + C + 2] ^= 0x40                 # one scale byte of row 13
    data[20 * (C + C // 8)] ^= 0x01                         # and a code byte of row 20
    out.write_bytes(bytes(data))
    bad = cli("verify", snap, out, "--chunk-rows", "4")
    assert bad.returncode == 1 and "MISMATCH row 13" in bad.stdout
    out.write_bytes(bytes(data[:-1]))                       # a truncated file never verifies
    assert cli("verify", snap, out).returncode in (1, 2)


def test_verify_refuses_a_manifest_for_another_checkpoint(tmp_path):
    snap, _ = make_ckpt(tmp_path)
    out = tmp_path / "m.rows"
    nr.build(snap, out)
    m = json.loads(Path(str(out) + ".json").read_text())
    m["shards"][1]["code_offset"] += 8
    Path(str(out) + ".json").write_text(json.dumps(m))
    r = cli("verify", snap, out)
    assert r.returncode == 2 and "manifest" in (r.stdout + r.stderr).lower()


def test_refusals(tmp_path):
    snap, _ = make_ckpt(tmp_path / "gap", drop=2)          # shard_2 missing: the row order would be wrong
    with pytest.raises(ValueError, match="shard"):
        nr.find_tables(snap)
    snap, _ = make_ckpt(tmp_path / "narrow", narrow=3)     # shards of different row widths
    with pytest.raises(ValueError, match="width"):
        nr.find_tables(snap)
    snap, _ = make_ckpt(tmp_path / "none", rows=(4, 4), spelling="part_{}")
    with pytest.raises(ValueError, match="no NVFP4 n-gram"):
        nr.build(snap, tmp_path / "x.rows")
