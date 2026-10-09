"""Thor's 48-row MXFP8 tiles (linear.thor_rows48): windows of 33-48 rows keep the 64-row tile's bits."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.kernels import qmm  # noqa: E402
from tensorfold.cuda.nvfp4 import linear as L  # noqa: E402


def _mx8(n: int, k: int, seed: int) -> L.Mx8Linear:
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn((n, k), generator=g, device="cuda") * 0.25).to(torch.float8_e4m3fn)
    s = torch.randint(118, 130, (n, k // 32), generator=g, device="cuda", dtype=torch.uint8)
    return L.Mx8Linear.from_checkpoint(w, s)


@pytest.mark.parametrize("n,k", [(16480, 2560), (2560, 6144), (1000, 512)])
def test_rows48_keep_the_64_row_bits(n, k, monkeypatch):
    lin = _mx8(n, k, 5)
    g = torch.Generator(device="cuda").manual_seed(13)
    for m in (33, 40, 44, 47, 48, 49, 64):
        x = (torch.randn((m, k), generator=g, device="cuda") * 0.5).to(torch.bfloat16)
        got = {}
        for flag in ("0", "1"):
            monkeypatch.setenv("TENSORFOLD_THOR_QMMF_ROWS48", flag)
            L.thor_rows48.cache_clear()
            got[flag] = lin(x)
        assert torch.equal(got["0"].view(torch.int16), got["1"].view(torch.int16)), (n, k, m, qmm.split_k(n, k))
        one = lin(x[m // 2:m // 2 + 1])                        # a row alone: the same bits as in the window
        assert torch.equal(one.view(torch.int16), got["1"][m // 2:m // 2 + 1].view(torch.int16)), (n, k, m)
    L.thor_rows48.cache_clear()


def test_rows48_switch_is_thor_only(monkeypatch):
    dev = torch.cuda.current_device()
    monkeypatch.setenv("TENSORFOLD_THOR_QMMF_ROWS48", "1")
    L.thor_rows48.cache_clear()
    assert L.thor_rows48(dev) == (torch.cuda.get_device_capability(dev) == (11, 0))
    monkeypatch.setenv("TENSORFOLD_THOR_QMMF_ROWS48", "0")
    L.thor_rows48.cache_clear()
    assert not L.thor_rows48(dev)
    L.thor_rows48.cache_clear()
