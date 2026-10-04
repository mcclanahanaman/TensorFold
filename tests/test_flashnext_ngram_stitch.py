"""Flash Next's n-gram row ids: the batch form and the prefix stitch Thor DEC-2's rows-ahead rely on (CPU, numpy only).

``NGram.ids(history, tokens)`` for a window equals the ids of a prefix followed by the ids of the rest with the last n-1
tokens of history + prefix as history (a token's ids depend on the n-1 tokens before it, an EOS among them resetting),
and ``rows_of`` gives every row of a batch the ids its own sequence would.
"""

import numpy as np
import pytest

from tensorfold.families.qwen4_exp.cuda.ngram import NGram

EOS = 0


def _ngram(n: int) -> NGram:
    return NGram(vocab=1000, ngram_size=n, heads_per_ngram=3, vocab_base=4000, divisor=8, shards=1, seed=11, eos=EOS,
                 embed_dim=96, ple_index=1)


def _seq(rng, size: int) -> np.ndarray:
    tokens = rng.integers(1, 1000, size=size)
    tokens[rng.random(size) < 0.2] = EOS                   # EOS often, so resets fall inside histories and windows
    return tokens.astype(np.int64)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_prefix_stitch_is_exact(n: int) -> None:
    g, rng = _ngram(n), np.random.default_rng(n)
    for _ in range(200):
        history, window = _seq(rng, n - 1), _seq(rng, int(rng.integers(1, 7)))
        full = g.ids(history, window)
        hist = history
        for k in range(len(window)):                        # one token at a time, as the MTP steps pick them
            np.testing.assert_array_equal(g.ids(hist, window[k:k + 1]), full[k:k + 1])
            hist = np.concatenate([hist, window[k:k + 1]])[-(n - 1):]
        for k in range(1, len(window)):                     # any prefix, then the rest in one call
            rest = g.ids(np.concatenate([history, window[:k]])[-(n - 1):], window[k:])
            np.testing.assert_array_equal(np.concatenate([g.ids(history, window[:k]), rest]), full)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_rows_of_batches_rows_independently(n: int) -> None:
    g, rng = _ngram(n), np.random.default_rng(10 + n)
    for width in (1, 2, 5):
        histories, windows = [_seq(rng, n - 1) for _ in range(6)], [_seq(rng, width) for _ in range(6)]
        rows = g.rows_of(np.stack([np.concatenate([h, t]) for h, t in zip(histories, windows)]))[:, -width:]
        for b, (h, t) in enumerate(zip(histories, windows)):
            np.testing.assert_array_equal(rows[b], g.ids(h, t))
