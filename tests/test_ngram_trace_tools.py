"""tools/thor/ngram_trace.py (FR-OPS-17, HD-2026-10-07-04 Phase 0): traffic -> exact PLE n-gram row ids -> cache-policy
replay. The contract the tool is written to (GLM card tk-ngram-trace); runs on any OS (the policy is injected here)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ngram_trace", ROOT / "tools" / "thor" / "ngram_trace.py")
nt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nt)


def encode(text: str) -> list[int]:
    return [ord(c) % 997 for c in text]


class LRU:
    """A reference policy with the NVMeTable.simulate contract: ids in order -> 1 where a lookup hit."""

    def __init__(self, rows: int) -> None:
        self.rows, self.d = rows, OrderedDict()

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


def test_session_rows_are_the_engine_hash():
    ng = nt.make_ngram()
    rows, src = nt.session_rows([("user", "ab"), ("assistant", "cd")], encode, ng)
    ids = encode("<|im_start|>user\nab<|im_end|>\n") + encode("<|im_start|>assistant\ncd<|im_end|>\n")
    assert rows.dtype == np.uint32 and rows.shape == (len(ids), 16)
    assert np.array_equal(rows, ng.ids(ng.initial_history(), np.asarray(ids, dtype=np.int64)).astype(np.uint32))
    n_user = len(encode("<|im_start|>user\nab<|im_end|>\n"))
    assert src.dtype == np.uint8 and src[:n_user].sum() == 0 and src[n_user:].all()


def test_jsonl_shapes():
    assert nt.jsonl_messages({"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]}) == \
        [("system", "s"), ("user", "u")]
    assert nt.jsonl_messages({"prompt": "p", "response": "r"}) == [("user", "p"), ("assistant", "r")]
    assert nt.jsonl_messages({"prompt": "p"}) == [("user", "p")]
    assert nt.jsonl_messages({"text": "t"}) == [("user", "t")]


def test_trace_round_trip(tmp_path):
    a = (np.arange(32, dtype=np.uint32).reshape(2, 16), np.array([0, 1], dtype=np.uint8))
    b = (np.arange(48, dtype=np.uint32).reshape(3, 16) + 7, np.array([1, 1, 0], dtype=np.uint8))
    summary = nt.write_trace(tmp_path, [("a", *a), ("b", *b)])
    assert summary["sessions"] == [{"name": "a", "tokens": 2, "assistant_tokens": 1},
                                   {"name": "b", "tokens": 3, "assistant_tokens": 2}]
    rows, src, sessions = nt.read_trace(tmp_path)
    assert np.array_equal(rows, np.concatenate([a[0], b[0]])) and np.array_equal(src, [0, 1, 1, 1, 0])
    assert [s["tokens"] for s in sessions] == [2, 3]


def test_interleave():
    assert nt.interleave([3, 5, 2], 2, 2).tolist() == [0, 1, 3, 4, 2, 5, 6, 8, 9, 7]
    assert nt.interleave([3, 5, 2], 1, 2).tolist() == list(range(10))
    order = nt.interleave([100, 37, 64, 5, 250], 3, 16)
    assert sorted(order.tolist()) == list(range(456))


def test_simulate_accounting():
    first = np.arange(16 * 50, dtype=np.uint32).reshape(50, 16)
    rows = np.concatenate([first, first])                  # the second half repeats the first exactly
    src = np.concatenate([np.zeros(50, np.uint8), np.ones(50, np.uint8)])
    r = nt.simulate(rows, src, np.arange(100), 10_000, 0, policy=lambda rows_, probation, init: LRU(rows_))
    assert r["lookups"] == 1600 and r["hit_all"] == 0.5
    assert r["hit_prompt"] == 0.0 and r["hit_generated"] == 1.0 and r["p_gen_token_miss"] == 0.0
    assert r["hit_bigram"] == 0.5 and r["hit_trigram"] == 0.5
    small = nt.simulate(rows, src, np.arange(100), 100, 0, policy=lambda rows_, probation, init: LRU(rows_))
    assert small["hit_all"] == 0.0 and small["p_gen_token_miss"] == 1.0


def test_piece_unique():
    rows = np.zeros((10, 16), dtype=np.uint32)
    rows[:, 0] = np.arange(10)                             # token t: ids {t, 0}
    got = nt.piece_unique(rows, [6, 4], 4)                 # pieces: s0 [0:4], [4:6]; s1 [0:4]
    assert got["pieces"] == 3 and got["max"] == 5 and got["mean"] == (4 + 3 + 5) / 3    # {0..3}, {4,5,0}, {6..9,0}


def test_cli_simulate(tmp_path):
    rows = np.arange(16 * 40, dtype=np.uint32).reshape(40, 16) % 300
    nt.write_trace(tmp_path, [("s0", rows[:25], np.zeros(25, np.uint8)), ("s1", rows[25:], np.ones(15, np.uint8))])
    out = tmp_path / "res.json"
    done = subprocess.run([sys.executable, str(ROOT / "tools" / "thor" / "ngram_trace.py"), "simulate", str(tmp_path),
                           "--streams", "2", "--chunk", "8", "--sizes", "0.0001,0.001", "--probation", "0",
                           "--policy", "lru", "--out", str(out)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    res = json.loads(out.read_text())
    cells = res["cells"]
    assert [(c["cache_rows"], c["probation"]) for c in cells] == [(100, 0), (1000, 0)]
    assert all(0.0 <= c["hit_all"] <= 1.0 for c in cells) and cells[1]["hit_all"] >= cells[0]["hit_all"]
    assert res["tokens"] == 40 and res["streams"] == 2
