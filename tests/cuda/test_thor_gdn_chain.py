"""Thor's decode chain kernel (gdn.cu ``chain_kernel``, DPUSH-5): every variant gives the tree kernel's bits for chain
windows: outputs, pending rows folded into committed states in place, and final states."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.kernels import gdn  # noqa: E402

thor = pytest.mark.skipif(torch.cuda.get_device_capability() != (11, 0), reason="Thor (sm_110) only")


def _rows(rows: int, hk: int, hv: int, dv: int, keys, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.nn.functional.normalize(torch.randn((rows, hk, 128), generator=gen, device="cuda"), dim=-1)
         * 128 ** -0.5).to(keys)
    k = torch.nn.functional.normalize(torch.randn((rows, hk, 128), generator=gen, device="cuda"), dim=-1).to(keys)
    v = torch.randn((rows, hv, dv), generator=gen, device="cuda").bfloat16()
    g = torch.rand((rows, hv), generator=gen, device="cuda") * 0.4 + 0.59
    beta = torch.rand((rows, hv), generator=gen, device="cuda")
    return q.contiguous(), k.contiguous(), v, g, beta


def _run(monkeypatch, variant, x, plan, states, pending, final=None):
    if variant is None:                                  # the default: the chain kernel from CHAIN_MIN_STREAMS streams
        monkeypatch.delenv(gdn.CHAIN_ENV, raising=False)
    else:
        monkeypatch.setenv(gdn.CHAIN_ENV, str(variant))
    st = [s.clone() for s in states]
    table = torch.tensor([s.data_ptr() for s in st], dtype=torch.int64, device="cuda")
    fin = None if final is None else [torch.zeros_like(s) for s in st]
    ftab = None if fin is None else torch.tensor([s.data_ptr() for s in fin], dtype=torch.int64, device="cuda")
    y = gdn.tree(*x, plan, table=table, pending=pending, final=ftab)
    torch.cuda.synchronize()
    return y, st, fin


def _same(a, b):
    ints = {torch.bfloat16: torch.int16, torch.float32: torch.int32}
    return torch.equal(a.view(ints[a.dtype]), b.view(ints[b.dtype]))


@thor
@pytest.mark.parametrize("keys", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("hk,hv,dv", [(16, 48, 128), (2, 6, 12)])
def test_chain_variants_keep_the_tree_kernels_bits(monkeypatch, keys, hk, hv, dv):
    windows, held = [3, 1, 7, 2, 5, 4, 6], [2, 0, 7, 1, 4, 3, 5]
    starts = [0]
    for w in windows:
        starts.append(starts[-1] + w)
    x = _rows(starts[-1], hk, hv, dv, keys, seed=11 + dv)
    plan = gdn.plan([list(range(-1, w - 1)) for w in windows], "cuda")
    assert plan.slots == 0
    gen = torch.Generator(device="cuda").manual_seed(5 + dv)
    states = [torch.randn((hv, dv, 128), generator=gen, device="cuda") * 0.05 for _ in windows]
    pk, pv, pg, pb = _rows(len(windows) * 7, hk, hv, dv, keys, seed=97 + dv)[1:]
    rows = torch.randperm(len(windows) * 7, generator=torch.Generator().manual_seed(3)).view(len(windows), 7)
    pending = (pk, pv, pg, pb, rows.to(torch.int32).cuda(), torch.tensor(held, dtype=torch.int32, device="cuda"))
    for final in (None, True):
        y0, st0, f0 = _run(monkeypatch, 0, x, plan, states, pending, final)
        assert not all(_same(a, b) for a, b in zip(st0, states))          # the pending rows were folded in
        for variant in [None, *range(1, gdn.CHAIN_VARIANTS + 1)]:
            y, st, fin = _run(monkeypatch, variant, x, plan, states, pending, final)
            assert _same(y, y0), (variant, final)
            assert all(_same(a, b) for a, b in zip(st, st0)), (variant, final)
            if final:
                assert all(_same(a, b) for a, b in zip(fin, f0)), variant


@thor
def test_chain_default_runs_from_six_streams_on_thor_and_zero_keeps_the_tree_kernel(monkeypatch):
    monkeypatch.delenv(gdn.CHAIN_ENV, raising=False)
    assert gdn.chain_variant("cuda", 8) == gdn.chain_variant("cuda", gdn.CHAIN_MIN_STREAMS) == gdn.CHAIN_DEFAULT > 0
    assert gdn.chain_variant("cuda", 4) == gdn.chain_variant("cuda", 1) == 0
    monkeypatch.setenv(gdn.CHAIN_ENV, "0")
    assert gdn.chain_variant("cuda", 8) == 0
    monkeypatch.setenv(gdn.CHAIN_ENV, "2")
    assert gdn.chain_variant("cuda", 1) == 2
    monkeypatch.setenv(gdn.CHAIN_ENV, str(gdn.CHAIN_VARIANTS + 1))
    with pytest.raises(ValueError):
        gdn.chain_variant("cuda", 8)


def test_other_gpus_keep_the_tree_kernel(monkeypatch):
    if torch.cuda.get_device_capability() == (11, 0):
        pytest.skip("not Thor")
    monkeypatch.setenv(gdn.CHAIN_ENV, "1")
    assert gdn.chain_variant("cuda", 8) == 0
