"""AGX Thor's prompt pieces: upstream's ``TENSORFOLD_PREFILL_ROWS`` (#238) sets them on every device; on Thor the
experts' scratch for a piece's rows is admitted with the window, the idle plan is the released one, live replies
stay bounded and wide pieces keep the pool guard."""

import pytest

from tensorfold.cuda.geometry import indexed_prefill_rows
from tensorfold.families.qwen4_exp.cuda.prompt_plan import choose, keeps_pool, pass_limit
from tests.test_flashnext_prompt_pieces import receipt, text

THOR = (11, 0)


@pytest.mark.parametrize("value", [None, "", "4096", "8192"])
def test_thor_keeps_the_released_idle_plan_and_upstreams_knob_sets_the_pieces(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("TENSORFOLD_PREFILL_ROWS", raising=False)
    else:
        monkeypatch.setenv("TENSORFOLD_PREFILL_ROWS", value)
    t = text()
    assert choose(receipt(t, 1 << 40), t, THOR) == (2048, 0)          # no Thor-only parser left in the idle plan
    rows = indexed_prefill_rows()
    assert rows == (int(value) if value else None)
    assert keeps_pool(rows or 2048, THOR) == bool(value)               # wide pieces keep the pool guard


def test_live_replies_keep_the_released_bound_beside_wide_pieces():
    assert pass_limit(8192, False, 0.0, None, None) == 8192
    assert pass_limit(8192, True, 0.0, None, None) == 2048
    assert 128 <= pass_limit(16384, True, 0.5, 0.05, 0.0001) <= 2048


@pytest.mark.torch
def test_wide_idle_pieces_and_live_bounds_in_the_decoder():
    from types import SimpleNamespace

    pytest.importorskip("torch")
    pytest.importorskip("triton")
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    dec = object.__new__(MultiDecoder)
    dec.prefill_rows, dec.share, dec.round_s, dec.row_s = 8192, 0.0, None, None
    dec.streams, dec.passed = {}, {}
    prompt = SimpleNamespace(sid=1, prompt=[1] * 20000, background=False)
    dec.filling = [prompt]
    dec.fills = {1: [SimpleNamespace(stops=[]), False, 0, None]}
    assert dec._pieces() == [(prompt, 0, 8192)]
    reply = SimpleNamespace(done=False)
    dec.streams[2] = reply
    assert dec._pieces() == [(prompt, 0, 2048)]
    reply.done = True
    assert dec._pieces() == [(prompt, 0, 8192)]


@pytest.mark.torch
def test_the_experts_scratch_by_config_matches_the_admission():
    pytest.importorskip("torch")
    from tensorfold.cuda.capacity import Geometry
    from tensorfold.cuda.nvfp4 import thor_experts

    t = text()
    by_rows = {rows: thor_experts.config_scratch_bytes(t, rows) for rows in (16, 2048, 4096, 8192, 16384)}
    assert by_rows[2048] == thor_experts.scratch_bytes(2048, 10, 512, 640, 2560)
    assert by_rows[16] < by_rows[2048] < by_rows[4096] < by_rows[8192] < by_rows[16384]
    base = Geometry(lambda slots: 1000 + slots, 4, 8)
    for prompt_rows, expect in ((None, 2048), (8192, 8192)):
        admitted = thor_experts.admission(lambda text: base, 16, prompt_rows)(t)
        assert admitted.bytes_at(100) - base.bytes_at(100) == 2 * by_rows[16] + by_rows[expect]
