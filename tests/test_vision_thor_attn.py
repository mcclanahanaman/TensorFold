"""THOR-VISION-ATTN: Qwen's vision-tower attention on Jetson Thor (sm_110) keeps transformers' bits."""

from types import SimpleNamespace

import pytest

from tensorfold.vision import thor_vision
from tensorfold.vision.qwen_cuda import QwenCudaVision


def _thor():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != thor_vision.CAPABILITY:
        pytest.skip("THOR-VISION-ATTN runs on sm_110 only")
    return torch


def _tower(torch, depth=2):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    config = Qwen3_5VisionConfig(depth=depth, hidden_size=1152, num_heads=16, intermediate_size=4304, patch_size=16,
                                 temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=2560, in_channels=3,
                                 num_position_embeddings=2304)
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return Qwen3_5VisionModel(config).to(device="cuda", dtype=torch.bfloat16).eval()


def test_other_gpus_and_the_switch_keep_transformers_forward(monkeypatch):
    tower = SimpleNamespace(blocks=[SimpleNamespace(attn=SimpleNamespace(forward="stale")) for _ in range(2)])
    monkeypatch.setattr(thor_vision, "available", lambda device=None: False)
    assert thor_vision.install(tower, "cuda") is None
    assert all("forward" not in vars(b.attn) for b in tower.blocks)       # a re-install starts from scratch


def test_the_master_switch_turns_the_path_off(monkeypatch):
    monkeypatch.setenv(thor_vision.ENV, "0")
    assert thor_vision.available() is False


def test_the_longest_segment_rides_the_tower_call_only_on_the_thor_path():
    vision = object.__new__(QwenCudaVision)
    vision._thor_attn = None
    assert vision._longest([1056, 480]) == {}
    vision._thor_attn = {"rope": True, "varlen": True}
    assert vision._longest([1056, 480]) == {"max_seqlen": 1056}


# video frame groups (equal segments), several images in one call (uneven), one image off the 128-row tile
@pytest.mark.parametrize("grids", [[[4, 24, 44]], [[1, 30, 40], [1, 16, 30], [1, 8, 12]], [[1, 18, 22]]])
@pytest.mark.parametrize("rope,varlen", [(True, True), (True, False), (False, True)])
def test_the_tower_keeps_transformers_bits(grids, rope, varlen):
    torch = _thor()
    tower = _tower(torch)
    patches = sum(t * h * w for t, h, w in grids)
    g = torch.Generator().manual_seed(7)
    pixels = (torch.randn(patches, 3 * 2 * 16 * 16, generator=g) * 0.5).to(device="cuda", dtype=torch.bfloat16)
    grid = torch.tensor(grids, dtype=torch.int64, device="cuda")
    longest = max(h * w for _, h, w in grids)
    with torch.inference_mode(), thor_vision.flash_kernels():
        want = tower(pixels, grid_thw=grid, return_dict=True)
        assert thor_vision.install(tower, "cuda", rope=rope, varlen=varlen) == {"rope": rope, "varlen": varlen}
        got = tower(pixels, grid_thw=grid, return_dict=True, max_seqlen=longest)
        synced = tower(pixels, grid_thw=grid, return_dict=True)              # no host length: one sync a layer
        assert thor_vision.install(tower, "cuda", rope=False, varlen=False) is None
        again = tower(pixels, grid_thw=grid, return_dict=True)
    for a, b in ((got, want), (synced, want), (again, want)):
        for name in ("last_hidden_state", "pooler_output"):
            x, y = getattr(a, name), getattr(b, name)
            assert x.dtype == y.dtype == torch.bfloat16 and torch.equal(x.view(torch.int16), y.view(torch.int16)), name


def test_the_load_check_refuses_different_bits():
    torch = _thor()
    attn = _tower(torch, depth=1).blocks[0].attn
    exact = thor_vision.make_forward(attn, thor_vision._rope_ext(), True)
    assert thor_vision.check(attn, exact, "cuda") is None

    def off_by_one(*args, **kwargs):
        out = exact(*args, **kwargs)
        return (out.view(torch.int16) ^ 1).view(torch.bfloat16)
    assert "differ" in thor_vision.check(attn, off_by_one, "cuda")
