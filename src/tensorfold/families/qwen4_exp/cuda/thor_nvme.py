"""Flash Next's PLE n-gram rows read from NVMe on Jetson Thor (sm_110) through a small row cache (FR-OPS-17).

HD-2026-10-07-04: the n-gram tables (26.8 GiB as NVFP4) stay on NVMe, neither locked in RAM nor in the page cache, so
that memory goes to the KV cache with the bf16 dense copies kept. ``NVMeTable`` keeps ``NVFP4Table``'s ``gather``
contract (row ids -> bf16 bits through the same pair table), so DEC-2, the prompt read-ahead and every consumer are
unchanged. Its rows come from ``thor_nvme.cpp`` (no Python per read): O_DIRECT reads of whole pages, Linux AIO from the
calling thread (pread threads where AIO is refused; io_uring is not used: Docker's default seccomp refuses it), and a
set-associative CLOCK cache of the checkpoint's 90-byte rows (80 code bytes + 10 block-scale bytes), allocated and
locked at load, before the KV pool measures its room. A row's bytes are the checkpoint's, and the bf16 values come from
the pair table ``NVFP4Table.use_pairs`` builds, so the rows equal the memory-mapped path's bit for bit.

Switches (read at load):
- ``TENSORFOLD_THOR_NGRAM_NVME``: ``1`` reads the tables from NVMe on sm_110; ``0`` or unset keeps the locked-table path
  (the kill switch). ``--ple-on-ssd`` with an NVFP4 checkpoint on sm_110 turns it on unless this is ``0``.
- ``TENSORFOLD_THOR_NGRAM_CACHE_GIB``: the row cache (default 0.53; HD-2026-10-07-04 allows 0.35-1.4 by measurement).
- ``TENSORFOLD_THOR_NGRAM_IO``: ``aio`` (default) or ``threads``; ``TENSORFOLD_THOR_NGRAM_DEPTH`` reads in flight a
  reader (64), ``TENSORFOLD_THOR_NGRAM_HELPERS`` extra reader threads for prompt-sized bursts (3),
  ``TENSORFOLD_THOR_NGRAM_PAGE`` bytes a read (4096), ``TENSORFOLD_THOR_NGRAM_PROBATION`` probation ways of 16 (4; 0:
  CLOCK over all ways, a new row starting with ``TENSORFOLD_THOR_NGRAM_INIT`` uses, 0).
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
import struct
import subprocess
from pathlib import Path

import numpy as np

from ..host_table import NVFP4Table, read_header

ENV = "TENSORFOLD_THOR_NGRAM_NVME"
CACHE_ENV = "TENSORFOLD_THOR_NGRAM_CACHE_GIB"
CAPABILITY = (11, 0)
SOURCE = Path(__file__).with_name("thor_nvme.cpp")
FLAGS = ("-O3", "-std=c++17", "-fPIC", "-shared", "-pthread")
ROW_SLOT = 95                     # cache bytes a row: 90 row bytes, a 4-byte tag, a use count
PROBATION = 4                     # probation ways of a set's 16 (TENSORFOLD_THOR_NGRAM_PROBATION; 0: CLOCK alone)
_LIB = None


def wanted(ple_on_ssd: bool = False) -> bool:
    """Whether the n-gram tables come from NVMe: sm_110 with the switch on (or ``--ple-on-ssd``, unless it is ``0``)."""

    flag = os.environ.get(ENV, "").strip().lower()
    if flag in ("0", "off", "false", "no") or not (flag in ("1", "on", "true", "yes") or ple_on_ssd):
        return False
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available() and tuple(torch.cuda.get_device_capability()) == CAPABILITY


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, "") or default)


def _build_dir() -> Path:
    root = os.environ.get("TENSORFOLD_THOR_NVME_EXT_DIR") or os.environ.get("TORCH_EXTENSIONS_DIR")
    return Path(root) / "tf_thor_nvme" if root else Path.home() / ".cache" / "tensorfold" / "thor_nvme"


def library():
    """The compiled reader (built once per source, under a lock, in the extensions directory)."""

    global _LIB
    if _LIB is not None:
        return _LIB
    cxx = os.environ.get("CXX", "c++")
    digest = hashlib.sha256(SOURCE.read_bytes() + " ".join((cxx,) + FLAGS).encode()).hexdigest()[:16]
    where = _build_dir()
    where.mkdir(parents=True, exist_ok=True)
    so = where / f"thor_nvme_{digest}.so"
    if not so.is_file():
        with open(where / "build.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not so.is_file():
                tmp = so.with_suffix(f".{os.getpid()}.tmp")
                done = subprocess.run([cxx, *FLAGS, "-o", str(tmp), str(SOURCE)], capture_output=True, text=True)
                if done.returncode != 0:
                    raise RuntimeError(f"building {SOURCE.name} failed: {done.stderr.strip()[-2000:]}")
                os.replace(tmp, so)
    lib = ctypes.CDLL(str(so))
    p, i64, i32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
    lib.tfnr_open.restype = p
    lib.tfnr_open.argtypes = (i32, ctypes.POINTER(ctypes.c_char_p), i32, p, i32, i32, i64, i32, i32, i32, i32, i32,
                              i32, i32, i32, ctypes.c_char_p, i32)
    lib.tfnr_close.argtypes = (p,)
    lib.tfnr_info.argtypes = lib.tfnr_stats.argtypes = (p, p)
    lib.tfnr_gather.restype = i32
    lib.tfnr_gather.argtypes = (p, p, i64, p, p, i32)
    lib.tfnr_records.restype = i64
    lib.tfnr_records.argtypes = (p, p, i64)
    lib.tfnr_reset_stats.argtypes = lib.tfnr_drop.argtypes = (p,)
    lib.tfnr_simulate.argtypes = (p, p, i64, p)
    lib.tfnr_background.argtypes = (p, i32)
    lib.tfnr_error.restype = ctypes.c_char_p
    lib.tfnr_error.argtypes = (p,)
    _LIB = lib
    return lib


STATS = ("calls", "lookups", "hits", "missed", "unique_misses", "reads", "bytes_read", "read_ns", "gather_ns", "errors",
         "inserted", "evicted")


class NVMeTable(NVFP4Table):
    """NVFP4 n-gram shards read from disk at each lookup through a row cache; ``gather`` as ``NVFP4Table``'s."""

    nvme = True

    def __init__(self, files: list[tuple[Path, dict, dict]], scale: float, *, cache_bytes: int | None = None,
                 direct: bool = True, aio: bool | None = None) -> None:
        from tensorfold.cuda.nvfp4.format import E2M1, e4m3

        paths: list[str] = []
        info: list[int] = []
        starts = [0]
        width = None
        for path, weight, block in files:
            if weight.get("dtype") != "U8" or block.get("dtype") != "F8_E4M3":
                raise ValueError(f"{Path(path).name}: NVFP4 n-gram shards are U8 codes with F8_E4M3 scales")
            rows, codes = (int(x) for x in weight["shape"])
            if width is None:
                width = codes
            if codes != width or list(block["shape"]) != [rows, codes // 8]:
                raise ValueError(f"{Path(path).name}: the n-gram shards differ in row width")
            if str(path) not in paths:
                paths.append(str(path))
            with open(path, "rb") as f:
                data = 8 + struct.unpack("<Q", f.read(8))[0]
            info += [paths.index(str(path)), data + int(weight["data_offsets"][0]), data + int(block["data_offsets"][0]),
                     rows]
            starts.append(starts[-1] + rows)
        self.starts = np.array(starts, dtype=np.int64)
        self.rows = int(self.starts[-1])
        self.wrow = int(width)
        self.width = 2 * self.wrow
        self.disk_bytes = self.rows * (self.wrow + self.wrow // 8)
        self.nbytes = 0                       # nothing mapped: the startup estimate and pool count only the cache
        self.e2m1, self.e4m3, self.g = E2M1, e4m3(np.arange(256)), np.float32(scale)
        self.pairs = None
        if cache_bytes is None:
            cache_bytes = int(float(os.environ.get(CACHE_ENV, "") or 0.53) * 2**30)
        io = os.environ.get("TENSORFOLD_THOR_NGRAM_IO", "aio").strip().lower()
        lib, err = library(), ctypes.create_string_buffer(512)
        names = (ctypes.c_char_p * len(paths))(*[p.encode() for p in paths])
        shard_info = np.asarray(info, dtype=np.int64)
        self._lib = lib
        self._h = lib.tfnr_open(len(paths), names, len(files), shard_info.ctypes.data, self.wrow, self.wrow // 8,
                                cache_bytes // ROW_SLOT, int(direct), int(io != "threads" if aio is None else aio),
                                _env_int("TENSORFOLD_THOR_NGRAM_DEPTH", 64), _env_int("TENSORFOLD_THOR_NGRAM_HELPERS", 3),
                                _env_int("TENSORFOLD_THOR_NGRAM_CPU", 8), _env_int("TENSORFOLD_THOR_NGRAM_PAGE", 4096),
                                _env_int("TENSORFOLD_THOR_NGRAM_INIT", 0),
                                _env_int("TENSORFOLD_THOR_NGRAM_PROBATION", PROBATION), err, len(err))
        if not self._h:
            raise OSError(f"n-gram rows from NVMe: {err.value.decode(errors='replace')}"
                          + (f" ({ENV}=0 keeps the tables in memory)" if direct else ""))
        self.use_pairs()

    @classmethod
    def policy(cls, cache_rows: int, probation: int = PROBATION, init: int = 0) -> "NVMeTable":
        """The cache policy alone, for trace studies (``simulate`` only: no table, no reads)."""

        self = cls.__new__(cls)
        self._lib, err = library(), ctypes.create_string_buffer(512)
        self._h = self._lib.tfnr_open(0, None, 0, None, 80, 10, int(cache_rows), 0, 1, 1, 0, 0, 4096, int(init),
                                      int(probation), err, len(err))
        if not self._h:
            raise OSError(err.value.decode(errors="replace"))
        return self

    def __del__(self) -> None:
        h, self._h = getattr(self, "_h", None), None
        if h:
            self._lib.tfnr_close(h)

    def info(self) -> dict:
        out = np.zeros(11, dtype=np.int64)
        self._lib.tfnr_info(self._h, out.ctypes.data)
        keys = ("rows", "cache_rows", "cache_bytes", "locked", "aio", "direct", "page", "readers", "cpu", "depth",
                "probation")
        return dict(zip(keys, (int(x) for x in out)))

    def note(self) -> str:
        i = self.info()
        return (f"n-gram rows from NVMe ({self.disk_bytes / 2**30:.2f} GiB on disk, O_DIRECT {i['page']}-byte reads, "
                f"{'AIO' if i['aio'] else 'pread threads'}, {1 + i['readers']} readers x {i['depth']} deep); row cache "
                f"{i['cache_rows'] / 1e6:.2f}M rows, {i['cache_bytes'] / 2**30:.2f} GiB, "
                f"{'locked' if i['locked'] else 'NOT locked'} ({ENV}=0: the tables in memory)")

    def _call(self, flat: np.ndarray, pairs, out: np.ndarray, kind: int) -> None:
        rc = self._lib.tfnr_gather(self._h, flat.ctypes.data, flat.size, None if pairs is None else pairs.ctypes.data,
                                   out.ctypes.data, kind)
        if rc == 34:                          # ERANGE
            raise ValueError(f"n-gram row ids must lie in [0, {self.rows})")
        if rc:
            raise OSError(rc, self._lib.tfnr_error(self._h).decode(errors="replace"))

    def raw(self, ids: np.ndarray) -> np.ndarray:
        """Rows ``ids`` as stored: [n, code bytes + scale bytes] uint8 (the code bytes, then the block scales)."""

        flat = np.ascontiguousarray(np.asarray(ids, dtype=np.int64).reshape(-1))
        out = np.empty((flat.size, self.wrow + self.wrow // 8), dtype=np.uint8)
        if flat.size:
            self._call(flat, None, out, 0)
        return out

    def gather(self, ids: np.ndarray, split: int | None = None) -> np.ndarray:
        """Rows ``ids`` (global) -> [n, W] uint16: the bf16 bits ``NVFP4Table.gather`` gives (``split`` unused)."""

        flat = np.ascontiguousarray(np.asarray(ids, dtype=np.int64).reshape(-1))
        if self.pairs is None:
            raw = self.raw(flat)
            return self.dequant(raw[:, :self.wrow], raw[:, self.wrow:])
        out = np.empty((flat.size, self.width), dtype=np.uint16)
        if flat.size:
            self._call(flat, self.pairs, out.view(np.uint32), 1)
        return out

    def lock(self) -> bool:
        return False

    def prefetch(self, workers: int = 8) -> float:
        return 0.0

    def populate(self, workers: int = 8) -> int:
        return 0

    def stats(self) -> dict:
        out = np.zeros(24, dtype=np.int64)
        self._lib.tfnr_stats(self._h, out.ctypes.data)
        return dict(zip(STATS, (int(x) for x in out)))

    def records(self, limit: int = 1 << 16) -> np.ndarray:
        """The last calls (up to 65,536): [rows, wall ns, read ns, unique misses] each."""

        out = np.zeros((limit, 4), dtype=np.int64)
        return out[:self._lib.tfnr_records(self._h, out.ctypes.data, limit)]

    def reset_stats(self) -> None:
        self._lib.tfnr_reset_stats(self._h)

    def drop(self) -> None:
        """Forget every cached row (the next lookups read from disk: a cold start for measurements)."""

        self._lib.tfnr_drop(self._h)

    def simulate(self, ids: np.ndarray) -> np.ndarray:
        """The cache policy alone over ``ids`` in order (no reads): 1 where a lookup hit."""

        flat = np.ascontiguousarray(np.asarray(ids, dtype=np.int64).reshape(-1))
        hit = np.zeros(flat.size, dtype=np.uint8)
        self._lib.tfnr_simulate(self._h, flat.ctypes.data, flat.size, hit.ctypes.data)
        return hit

    def background(self, threads: int) -> None:
        """Benchmarks: ``threads`` readers of random pages until called again with 0."""

        self._lib.tfnr_background(self._h, int(threads))


def open_table(model_dir: Path, shards: list[tuple[str, str]], scale, **kwargs) -> NVMeTable:
    """The NVFP4 n-gram table of ``shards`` ((file, tensor key) pairs) read from NVMe; other layouts are refused."""

    headers: dict[str, dict] = {}
    files = []
    for shard, key in shards:
        if shard not in headers:
            headers[shard] = read_header(Path(model_dir) / shard)
        h = headers[shard]
        if key + ".weight_scale" not in h or h[key + ".weight"].get("dtype") != "U8":
            raise ValueError(f"{ENV}: only NVFP4 n-gram shards are read from NVMe ({key} is not); set {ENV}=0")
        files.append((Path(model_dir) / shard, h[key + ".weight"], h[key + ".weight_scale"]))
    return NVMeTable(files, scale("weight_scale_2"), **kwargs)


__all__ = ["CACHE_ENV", "ENV", "NVMeTable", "library", "open_table", "wanted"]
