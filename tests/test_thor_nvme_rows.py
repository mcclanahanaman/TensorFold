"""thor_nvme (FR-OPS-17, HD-2026-10-07-04): NVFP4 n-gram rows read from disk through the native row cache equal the
memory-mapped NVFP4Table's bit for bit: raw bytes, pair-table gathers and per-call dequant, with and without the cache,
AIO and pread threads, O_DIRECT where the file system allows it, duplicates, evictions and concurrent callers."""

from __future__ import annotations

import json
import os
import struct
import sys
import threading

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the NVMe reader is Linux-only")

from tensorfold.families.qwen4_exp.cuda import thor_nvme  # noqa: E402
from tensorfold.families.qwen4_exp.host_table import NVFP4Table, read_header  # noqa: E402

SHARDS = ((0, 1000), (0, 777), (1, 1234), (1, 513))     # (file, rows)
CODES = 80


def _write(path, tensors) -> None:
    header, blobs, at = {"__metadata__": {"format": "pt"}}, [], 0
    for name, dtype, array in tensors:
        data = np.ascontiguousarray(array).tobytes()
        header[name] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [at, at + len(data)]}
        blobs.append(data)
        at += len(data)
    text = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)) + text + b"".join(blobs))


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("ngram")
    rng = np.random.default_rng(7)
    files: dict[int, list] = {0: [], 1: []}
    keys = []
    for i, (f, rows) in enumerate(SHARDS):
        key = f"model.layers.1.ple_embedding.ngram_embedding.shard_{i}"
        files[f].append((key + ".weight", "U8", rng.integers(0, 256, (rows, CODES), dtype=np.uint8)))
        files[f].append((key + ".weight_scale", "F8_E4M3", rng.integers(0, 256, (rows, CODES // 8), dtype=np.uint8)))
        keys.append((f"model-{f}.safetensors", key))
    for f, tensors in files.items():    # an odd-sized tensor first: the shards start off page boundaries
        _write(root / f"model-{f}.safetensors", [("lead", "U8", rng.integers(0, 256, (37,), dtype=np.uint8))] + tensors)
    triples = []
    for name, key in keys:
        h = read_header(root / name)
        triples.append((root / name, h[key + ".weight"], h[key + ".weight_scale"]))
    return root, keys, triples


def _direct_ok(path) -> bool:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    except OSError:
        return False
    os.close(fd)
    return True


def _tables(triples, cache_rows: int, aio: bool):
    ref = NVFP4Table(triples, 0.03125)
    direct = _direct_ok(triples[0][0])
    got = thor_nvme.NVMeTable(triples, 0.03125, cache_bytes=cache_rows * thor_nvme.ROW_SLOT, direct=direct, aio=aio)
    return ref, got


@pytest.mark.parametrize("cache_rows", [0, 64, 100_000])
@pytest.mark.parametrize("aio", [True, False])
def test_rows_equal_the_mapped_table(checkpoint, cache_rows, aio):
    _, _, triples = checkpoint
    ref, got = _tables(triples, cache_rows, aio)
    rng = np.random.default_rng(cache_rows + aio)
    ids = np.concatenate([rng.integers(0, ref.rows, 3000), np.arange(ref.rows - 5, ref.rows), np.arange(0, 5),
                          ref.starts[1:-1], ref.starts[1:-1] - 1])                  # shard edges and both ends
    ids = np.concatenate([ids, ids[:500]])                                          # duplicates in one call
    shard, local = ref._where(ids)
    raw = got.raw(ids)
    for f in range(len(triples)):
        at = shard == f
        assert np.array_equal(raw[at, :CODES], ref.values[f][local[at]])
        assert np.array_equal(raw[at, CODES:], ref.scales[f][local[at]])
    plain = ref.gather(ids)                     # per-call dequant
    ref.use_pairs()
    assert np.array_equal(ref.gather(ids), plain)
    for _ in range(3):                          # misses, then hits (or evictions with the small cache)
        assert np.array_equal(got.gather(ids), plain)
        two_d = ids[:1600].reshape(100, 16)
        assert np.array_equal(got.gather(two_d), plain[:1600])
    got.pairs = None                            # the per-call dequant path over raw rows
    assert np.array_equal(got.gather(ids), plain)
    s = got.stats()
    assert s["lookups"] == 5 * ids.size + 3 * 1600 and s["errors"] == 0
    if cache_rows >= 100_000:
        assert s["hits"] > s["lookups"] // 2 and s["evicted"] == 0
    if cache_rows == 0:
        assert s["hits"] == 0


def test_edges_and_errors(checkpoint):
    _, _, triples = checkpoint
    ref, got = _tables(triples, 1000, True)
    assert got.gather(np.zeros(0, dtype=np.int64)).shape == (0, 2 * CODES)
    for bad in ([-1], [ref.rows]):
        with pytest.raises(ValueError):
            got.gather(np.asarray(bad))
    assert got.rows == ref.rows and got.width == ref.width and got.nbytes == 0 and got.disk_bytes == ref.nbytes
    assert got.lock() is False and got.populate() == 0
    info = got.info()
    assert info["rows"] == ref.rows and info["cache_rows"] >= 1000


def test_concurrent_callers(checkpoint):
    _, _, triples = checkpoint
    ref, got = _tables(triples, 300, True)
    ref.use_pairs()
    errors = []

    def worker(seed: int) -> None:
        rng = np.random.default_rng(seed)
        for _ in range(20):
            ids = rng.integers(0, ref.rows, int(rng.integers(1, 6000)))
            if not np.array_equal(got.gather(ids), ref.gather(ids)):
                errors.append(seed)

    threads = [threading.Thread(target=worker, args=(s,)) for s in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and got.stats()["errors"] == 0


def test_policy_drop_and_records(checkpoint):
    _, _, triples = checkpoint
    _, got = _tables(triples, 4096, True)
    ids = np.arange(0, 2000, dtype=np.int64)
    assert got.simulate(ids).sum() == 0 and got.simulate(ids).sum() == 2000    # a second pass hits (fits)
    got.drop()
    assert got.simulate(ids[:10]).sum() == 0
    got.drop()
    got.reset_stats()
    got.gather(ids)
    got.gather(ids)
    rec = got.records()
    assert rec.shape == (2, 4) and rec[0, 0] == 2000 and rec[0, 3] == 2000 and rec[1, 3] == 0
    s = got.stats()
    assert s["calls"] == 2 and s["hits"] == 2000 and s["unique_misses"] == 2000 and s["reads"] == 4000


def test_scan_resistance(checkpoint):
    """Rows used twice outlive a one-pass scan of new rows (a new row starts with no uses)."""

    _, _, triples = checkpoint
    _, got = _tables(triples, 1024, True)
    hot = np.arange(0, 256, dtype=np.int64)
    got.simulate(hot)
    got.simulate(hot)                           # used twice
    got.simulate(np.arange(1000, 3500, dtype=np.int64))    # a scan of 2,500 new rows through a 1,024-row cache
    assert got.simulate(hot).mean() > 0.9


def test_wanted_needs_the_switch_and_sm110(monkeypatch):
    monkeypatch.delenv(thor_nvme.ENV, raising=False)
    assert thor_nvme.wanted() is False
    monkeypatch.setenv(thor_nvme.ENV, "0")
    assert thor_nvme.wanted(ple_on_ssd=True) is False


def test_policy_only_table():
    sim = thor_nvme.NVMeTable.policy(4096, probation=0)
    ids = np.arange(0, 3000, dtype=np.int64)
    assert sim.simulate(ids).sum() == 0 and sim.simulate(ids).sum() == 3000
    assert thor_nvme.NVMeTable.policy(4096).info()["probation"] == thor_nvme.PROBATION
