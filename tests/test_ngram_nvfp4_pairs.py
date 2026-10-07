"""NVFP4 n-gram rows through the pair table (Thor, sm_110): the same uint16 bits as dequantizing each call's rows."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from tensorfold.families.qwen4_exp import host_table
from tensorfold.families.qwen4_exp.host_table import NVFP4Table, open_table

# table scales: a checkpoint-like one, one with every fp32 mantissa bit, tiny (products under bf16's normal range),
# huge (products overflow to inf), negative
SCALES = (0.0371, float(np.float32(1 / 3)), 2.0 ** -133, 3.0e38, -0.0117)


def _table(tmp_path, shards, g: float, tag: str) -> NVFP4Table:
    """An NVFP4 table over ``shards`` [(codes [r, C] uint8, block scales [r, C / 8] uint8)] with table scale ``g``."""

    folder = tmp_path / tag
    folder.mkdir()
    keys = []
    for i, (codes, scales) in enumerate(shards):
        blobs = {f"t.shard_{i}.weight": ("U8", codes), f"t.shard_{i}.weight_scale": ("F8_E4M3", scales)}
        header, data, at = {}, b"", 0
        for name, (dtype, arr) in blobs.items():
            header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [at, at + arr.nbytes]}
            data, at = data + arr.tobytes(), at + arr.nbytes
        raw = json.dumps(header).encode()
        raw += b" " * (-len(raw) % 8)
        (folder / f"s{i}.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + data)
        keys.append((f"s{i}.safetensors", f"t.shard_{i}"))
    table = open_table(folder, keys, lambda name: g)
    assert isinstance(table, NVFP4Table)
    return table


@pytest.mark.parametrize("g", SCALES)
def test_every_scale_and_code_byte_gives_the_dequantized_bits(tmp_path, g):
    byte = np.arange(256, dtype=np.uint8)
    shard = (np.tile(byte, (256, 1)), np.repeat(byte[:, None], 32, axis=1))     # row s: every code byte under scale s
    ref, fast = _table(tmp_path, [shard], g, "ref"), _table(tmp_path, [shard], g, "fast")
    fast.use_pairs()
    assert fast.pairs.shape == (65536,) and fast.pairs.dtype == np.uint32
    ids = np.arange(256)
    want, got = ref.gather(ids), fast.gather(ids)
    assert got.dtype == np.uint16 and got.shape == (256, 512)
    assert np.array_equal(got, want)
    assert ref.pairs is None and np.isnan(ref.e4m3[0x7F]) and (got[0x7F] == want[0x7F]).all()   # NaN scales too


@pytest.mark.parametrize("split", [None, 32])
def test_random_rows_one_to_twenty_thousand_match_bit_for_bit(tmp_path, monkeypatch, split):
    rng = np.random.default_rng(7)
    shards = [(rng.integers(0, 256, size=(r, 80), dtype=np.uint8), rng.integers(0, 256, size=(r, 10), dtype=np.uint8))
              for r in (9001, 5003, 7000)]
    g = SCALES[0]
    ref, fast = _table(tmp_path, shards, g, "ref"), _table(tmp_path, shards, g, "fast")
    fast.use_pairs()
    for n in (1, 2, 3, 15, 16, 17, 31, 32, 33, 63, 64, 65, 383, 384, 385, 1023, 1024, 1025, 2047, 2048, 2049, 4096,
              8191, 12345, 20000):
        ids = rng.integers(0, ref.rows, size=n)
        want, got = ref.gather(ids, split), fast.gather(ids, split)
        assert got.shape == (n, 160) and np.array_equal(got, want), n
    heads = rng.integers(0, ref.rows, size=(1250, 16))                       # the engine's [tokens, heads] ids
    assert np.array_equal(fast.gather(heads, split), ref.gather(heads, split))
    monkeypatch.setattr(host_table, "PAIR_ROWS", 7)                          # pieces split across the bound
    ids = rng.integers(0, ref.rows, size=5000)
    assert np.array_equal(fast.gather(ids, split), ref.gather(ids, split))
    assert fast.gather(np.zeros(0, dtype=np.int64), split).shape == (0, 160)
    with pytest.raises(ValueError):
        fast.gather(np.array([ref.rows]))


@pytest.mark.torch
def test_thor_setup_turns_the_pair_table_on_and_its_switch_off(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import thor_dec2

    for name in ("batch", "split", "pool_rows", "ahead", "tables", "check"):
        monkeypatch.setattr(thor_dec2.S, name, getattr(thor_dec2.S, name))
    rng = np.random.default_rng(3)
    shard = (rng.integers(0, 256, size=(40, 80), dtype=np.uint8), rng.integers(0, 256, size=(40, 10), dtype=np.uint8))
    on, off = _table(tmp_path, [shard], 0.05, "on"), _table(tmp_path, [shard], 0.05, "off")
    note = thor_dec2.setup([on], locked=True)
    assert on.pairs is not None and "pair table" in note
    monkeypatch.setenv(thor_dec2.PAIRS_ENV, "0")
    note = thor_dec2.setup([off], locked=True)
    assert off.pairs is None and "pair table" not in note
    ids = rng.integers(0, 40, size=300)
    assert np.array_equal(on.gather(ids), off.gather(ids))
