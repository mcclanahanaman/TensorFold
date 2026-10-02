"""AGX Thor's idle prompt pieces (``TENSORFOLD_PREFILL_ROWS``): the released 2048 rows unless asked, every extra byte
counted (the Thor experts' scratch too), the admitted window kept, live replies bounded, other devices untouched."""

import copy

import pytest

from tensorfold.cuda.geometry import indexed_prompt_bytes
from tensorfold.families.qwen4_exp.cuda import prompt_plan
from tensorfold.families.qwen4_exp.cuda.prompt_plan import ROWS_ENV, choose, pass_limit
from tests.test_flashnext_prompt_pieces import receipt, text

THOR = (11, 0)


def scratch(t, rows):            # a stand-in for the Thor path's own bytes: grows with the rows, not linearly
    return 1000 * rows + 7 * rows * rows // 64


def extra(t, rows):
    return indexed_prompt_bytes(t, rows - 2048) + scratch(t, rows) - scratch(t, 2048)


@pytest.mark.parametrize("value", [None, "", " ", "2048"])
def test_unset_or_released_rows_keep_todays_plan(monkeypatch, capsys, value):
    if value is None:
        monkeypatch.delenv(ROWS_ENV, raising=False)
    else:
        monkeypatch.setenv(ROWS_ENV, value)
    t = text()
    plan = receipt(t, 1 << 40)
    old = copy.deepcopy(plan)
    assert choose(plan, t, THOR, thor=scratch) == (2048, 0)
    assert choose(plan, t, THOR) == (2048, 0)
    assert plan == old
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("rows", [2112, 4096, 8192, 16384])
@pytest.mark.parametrize("loading", [0, 1 << 40])
def test_wider_pieces_count_every_extra_byte_without_reducing_the_window(monkeypatch, rows, loading):
    monkeypatch.setenv(ROWS_ENV, str(rows))
    t = text()
    more = extra(t, rows)
    plan = receipt(t, more, loading=loading)
    old = copy.deepcopy(plan)
    got, workspace = choose(plan, t, THOR, thor=scratch)
    assert got == rows and workspace == indexed_prompt_bytes(t, rows)
    assert plan["prefill_rows"] == rows and plan["prompt_workspace_bytes_estimate"] == workspace
    for key in ("context_window", "cache_slots", "largest_window", "budget_bytes", "startup_peak_bytes_estimate",
                "mapped_table_bytes"):
        assert plan[key] == old[key]
    assert plan["cache_workspace_bytes_estimate"] == old["cache_workspace_bytes_estimate"] + more
    assert plan["serving_peak_bytes_estimate"] == old["serving_peak_bytes_estimate"] + more
    peak = max(plan["serving_peak_bytes_estimate"], old["startup_peak_bytes_estimate"])
    assert plan["total_bytes_estimate"] == peak <= plan["budget_bytes"]
    assert plan["full_mapped_working_set_bytes_estimate"] == peak + plan["mapped_table_bytes"]


def test_the_thor_scratch_is_part_of_the_extra_bytes(monkeypatch, capsys):
    monkeypatch.setenv(ROWS_ENV, "8192")
    t = text()
    plan = receipt(t, indexed_prompt_bytes(t, 8192 - 2048))      # room for the prompt buffers, not the scratch
    old = copy.deepcopy(plan)
    assert choose(plan, t, THOR, thor=scratch) == (2048, 0)
    assert plan == old
    assert choose(plan, t, THOR, thor=lambda t, rows: 0)[0] == 8192   # with no Thor scratch it fits


def test_a_budget_without_room_keeps_released_pieces_and_says_why(monkeypatch, capsys):
    monkeypatch.setenv(ROWS_ENV, "8192")
    t = text()
    plan = receipt(t, extra(t, 8192) - 1)
    old = copy.deepcopy(plan)
    assert choose(plan, t, THOR, thor=scratch) == (2048, 0)
    assert plan == old
    out = capsys.readouterr().out
    assert f"{ROWS_ENV}=8192" in out and "budget" in out and "stay 2048 rows" in out


@pytest.mark.parametrize("value", ["1024", "2047", "2100", "4095", "16448", "32768", "0", "-4096", "4096.0", "8k",
                                   "abc"])
def test_values_off_the_grid_or_out_of_range_are_refused(monkeypatch, value):
    monkeypatch.setenv(ROWS_ENV, value)
    t = text()
    with pytest.raises(ValueError, match=ROWS_ENV):
        choose(receipt(t, 1 << 40), t, THOR, thor=scratch)


@pytest.mark.parametrize("world,vision,fp8,thor", [(2, False, False, scratch), (1, True, False, scratch),
                                                   (1, False, True, scratch), (1, False, False, None)])
def test_two_gpus_images_fp8_or_another_checkpoint_keep_released_pieces(monkeypatch, capsys, world, vision, fp8,
                                                                         thor):
    monkeypatch.setenv(ROWS_ENV, "8192")
    t = text()
    plan = receipt(t, 1 << 40)
    old = copy.deepcopy(plan)
    assert choose(plan, t, THOR, world=world, vision=vision, fp8=fp8, thor=thor) == (2048, 0)
    assert plan == old
    assert "stay 2048 rows" in capsys.readouterr().out


@pytest.mark.parametrize("capability,expected", [((12, 1), 4096), ((12, 0), 2048), ((10, 0), 2048), ((9, 0), 2048)])
def test_other_devices_ignore_the_switch(monkeypatch, capsys, capability, expected):
    monkeypatch.setenv(ROWS_ENV, "8192")
    t = text()
    plan = receipt(t, 1 << 40)
    assert choose(plan, t, capability, thor=scratch)[0] == expected     # GB10 keeps its own 4096-row rule
    monkeypatch.setenv(ROWS_ENV, "not a number")                          # never read off Thor
    assert choose(receipt(t, 1 << 40), t, capability, thor=scratch)[0] == expected
    assert capsys.readouterr().out == ""


def test_requested_reads_the_switch(monkeypatch):
    monkeypatch.delenv(ROWS_ENV, raising=False)
    assert prompt_plan.requested() == 2048
    for rows in range(2048, prompt_plan.MAX_ROWS + 1, 64):
        monkeypatch.setenv(ROWS_ENV, f" {rows} ")
        assert prompt_plan.requested() == rows


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
