"""Thor's MXFP8 slice path (linear.thor_slices): K slices met through the reduce keep the cluster's bits at every row count."""

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


@pytest.mark.parametrize("n,k,sk", [(2560, 6144, 8), (2560, 2560, 4)])
def test_slices_keep_the_cluster_bits(n, k, sk, monkeypatch):
    lin = _mx8(n, k, 7)
    assert qmm.split_k(n, k) == sk
    g = torch.Generator(device="cuda").manual_seed(11)
    for m in (1, 2, 5, 7, 16, 33, 64, 128, 255):
        x = (torch.randn((m, k), generator=g, device="cuda") * 0.5).to(torch.bfloat16)
        got = {}
        for flag in ("0", "1"):
            monkeypatch.setenv("TENSORFOLD_THOR_QMMF_SLICES", flag)
            L.thor_slices.cache_clear()
            got[flag] = lin(x)
        assert torch.equal(got["0"], got["1"]), m
    L.thor_slices.cache_clear()


def test_slices_switch_is_thor_only(monkeypatch):
    dev = torch.cuda.current_device()
    monkeypatch.setenv("TENSORFOLD_THOR_QMMF_SLICES", "1")
    L.thor_slices.cache_clear()
    assert L.thor_slices(dev) == (torch.cuda.get_device_capability(dev) == (11, 0))
    monkeypatch.setenv("TENSORFOLD_THOR_QMMF_SLICES", "0")
    L.thor_slices.cache_clear()
    assert not L.thor_slices(dev)
    L.thor_slices.cache_clear()
