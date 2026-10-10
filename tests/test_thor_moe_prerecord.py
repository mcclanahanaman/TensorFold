"""Jetson Thor (sm_110): a decode buffer set's routed-expert graphs recorded at load (``nvfp4_moe.prerecord_thor``).
Every (layer, window width) the set's calls can make goes through ``forward.moe_block``'s own
call (its rows, router, plan and slots) before any request, so no request pays a capture; a set without graphs, or
the switch off, records nothing. CPU fakes: the call pattern and the accounting, not the device work."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.cuda import nvfp4_moe  # noqa: E402


def _fakes(monkeypatch, rows=5, graphs=True, prefill=False):
    calls = []
    scratch = SimpleNamespace(rows=rows, graphs={} if graphs else None)
    b = SimpleNamespace(mixed=torch.ones(8, 3), xs_mixed=torch.ones(8, 3), moe=SimpleNamespace(thor=scratch),
                        prefill=prefill)

    def fake_moe(x, xs, router, experts, buf, cfg, *, unscattered=False):
        assert buf is b.moe and unscattered is prefill
        assert x.data_ptr() == b.mixed.data_ptr() and xs.data_ptr() == b.xs_mixed.data_ptr()   # moe_block's rows
        assert float(x.abs().sum()) == 0.0                                                  # zeroed first
        assert cfg.num_experts_per_tok == 10 and experts == "e" + router[1:]
        calls.append((x.shape[0], router))
        scratch.graphs[(router, x.shape[0])] = object()          # thor.run records a key on its first call

    free = iter([1000 << 20, 600 << 20])
    monkeypatch.setattr(nvfp4_moe, "moe", fake_moe)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (next(free), 2000 << 20))
    layers = [SimpleNamespace(moe=SimpleNamespace(router=f"r{i}", experts=f"e{i}")) for i in range(3)]
    w = SimpleNamespace(cfg=SimpleNamespace(top_k=10, experts=256, moe_width=512, hidden=2048), comm=None)
    return calls, scratch, b, layers, w


def test_every_layer_at_every_window_width_is_recorded_at_load(monkeypatch):
    monkeypatch.delenv("TENSORFOLD_THOR_FI_MOE_PRERECORD", raising=False)
    calls, scratch, b, layers, w = _fakes(monkeypatch, rows=5)
    n, took = nvfp4_moe.prerecord_thor(w, b, layers)
    assert calls == [(r, f"r{i}") for r in range(1, 6) for i in range(3)]
    assert (n, took) == (15, 400 << 20) and len(scratch.graphs) == 15
    assert float(b.mixed[5:].sum()) == 3 * 3                    # rows past the scratch's are left alone


def test_a_set_without_graphs_or_the_switch_off_records_nothing(monkeypatch):
    calls, scratch, b, layers, w = _fakes(monkeypatch, graphs=False)
    assert nvfp4_moe.prerecord_thor(w, b, layers) == (0, 0) and not calls
    calls, scratch, b, layers, w = _fakes(monkeypatch)
    monkeypatch.setenv("TENSORFOLD_THOR_FI_MOE_PRERECORD", "0")
    assert nvfp4_moe.prerecord_thor(w, b, layers) == (0, 0) and not calls and scratch.graphs == {}
    monkeypatch.delenv("TENSORFOLD_THOR_FI_MOE_PRERECORD")
    assert nvfp4_moe.prerecord_thor(w, b, []) == (0, 0) and not calls
