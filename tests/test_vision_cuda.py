"""CUDA image admission, checkpoint and transport contracts without accelerator runtimes."""

import ast
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tensorfold.vision.qwen_cuda import (EncodedVision, broadcast_encoded, capacity_geometry,
                                       checkpoint_vision, validate_encoded, weight_transform)


def _checkpoint(path):
    vision = {"model_type": "qwen3_5", "hidden_size": 8, "out_hidden_size": 8, "depth": 1,
              "patch_size": 2, "temporal_patch_size": 2, "spatial_merge_size": 2, "in_channels": 3,
              "intermediate_size": 12, "num_heads": 2, "num_position_embeddings": 4}
    config = {"model_type": "qwen3_5", "vision_config": vision, "text_config": {
        "hidden_size": 8, "head_dim": 8, "rope_parameters": {"mrope_interleaved": True,
        "mrope_section": [2, 1, 1], "partial_rotary_factor": 1}}}
    (path / "config.json").write_text(json.dumps(config))
    shapes = {"patch_embed.proj.weight": [8, 2, 2, 2, 3], "patch_embed.proj.bias": [8],
              "pos_embed.weight": [4, 8], "merger.norm.weight": [8], "merger.norm.bias": [8],
              "merger.linear_fc1.weight": [32, 32], "merger.linear_fc1.bias": [32],
              "merger.linear_fc2.weight": [8, 32], "merger.linear_fc2.bias": [8]}
    for part, shape in {"norm1": [8], "norm2": [8], "attn.qkv": [24, 8], "attn.proj": [8, 8],
                        "mlp.linear_fc1": [12, 8], "mlp.linear_fc2": [8, 12]}.items():
        shapes[f"blocks.0.{part}.weight"] = shape
        shapes[f"blocks.0.{part}.bias"] = [shape[0]]
    offset, entries = 0, {}
    for name, shape in shapes.items():
        size = int(np.prod(shape)) * 2
        entries["vision_tower." + name] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    _write_tensors(path, entries, offset)
    return entries, offset


def _write_tensors(path, entries, size):
    raw = json.dumps(entries).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(size))


def test_vision_headers_do_not_load_tensor_payloads(tmp_path):
    _, size = _checkpoint(tmp_path)
    config, resident = checkpoint_vision(tmp_path)
    assert config["out_hidden_size"] == 8
    assert resident == size


@pytest.mark.parametrize("damage", ["missing", "quantized", "range", "shape"])
def test_incomplete_or_incompatible_towers_refuse_before_loading(tmp_path, damage):
    entries, size = _checkpoint(tmp_path)
    key = "vision_tower.blocks.0.attn.qkv.weight"
    if damage == "missing":
        del entries[key]
    elif damage == "quantized":
        entries[key]["dtype"] = "U32"
    elif damage == "range":
        entries[key]["data_offsets"] = [size, size + 24 * 8 * 2]
    else:
        entries[key]["shape"] = [12, 16]
    _write_tensors(tmp_path, entries, size)
    with pytest.raises(ValueError):
        checkpoint_vision(tmp_path)


def test_every_placeholder_has_exactly_one_feature_and_position():
    prompt = [10, 99, 99, 99, 99, 11, 12]
    positions = [[0, 1, 1, 1, 1, 3, 4], [0, 1, 1, 2, 2, 3, 4], [0, 1, 2, 1, 2, 3, 4]]
    validate_encoded((1, 2, 3, 4), positions, -2, prompt, 99, (4, 8), 8)
    for rows, pos, delta, shape in [((0, 1, 2, 3), positions, -2, (4, 8)),
                                   ((1, 2, 3, 4), positions, 0, (4, 8)),
                                   ((1, 2, 3, 4), [positions[0]] * 2, -2, (4, 8)),
                                   ((1, 2, 3, 4), positions, -2, (3, 8))]:
        with pytest.raises(ValueError):
            validate_encoded(rows, pos, delta, prompt, 99, shape, 8)


def test_vision_memory_is_reserved_only_on_the_tower_rank(tmp_path):
    from tensorfold.cuda.capacity import Geometry

    _checkpoint(tmp_path)
    base = lambda text: Geometry(lambda slots: slots * 64, 8)
    zero = capacity_geometry(base, tmp_path, True, 0)({})
    one = capacity_geometry(base, tmp_path, True, 1)({})
    plain = capacity_geometry(base, tmp_path, False, 0)({})
    assert zero.needed(32) > one.needed(32) > plain.needed(32)
    original = lambda *args: (0, 0)
    info = {"shape": [8, 8], "dtype": "BF16"}
    assert weight_transform(original, True, 0)("vision_tower.x", info) == (128, 0)
    assert weight_transform(original, True, 1)("vision_tower.x", info) == (0, 0)


def test_offloaded_tower_leaves_the_gpu_budget_but_keeps_a_smaller_workspace(tmp_path):
    from tensorfold.cuda.capacity import Geometry
    from tensorfold.vision.qwen_cuda import OFFLOAD_WORKSPACE_BYTES, WORKSPACE_BYTES

    _checkpoint(tmp_path)
    base = lambda text: Geometry(lambda slots: slots * 64, 8)
    resident = capacity_geometry(base, tmp_path, True, 0)({})
    offloaded = capacity_geometry(base, tmp_path, True, 0, offload=True)({})
    plain = capacity_geometry(base, tmp_path, False, 0)({})
    assert resident.needed(32) > offloaded.needed(32) > plain.needed(32)
    assert offloaded.needed(32) - plain.needed(32) == OFFLOAD_WORKSPACE_BYTES < WORKSPACE_BYTES
    original = lambda *args: (0, 0)
    info = {"shape": [8, 8], "dtype": "BF16"}
    assert weight_transform(original, True, 0, True)("vision_tower.x", info) == (0, 0)
    assert weight_transform(original, True, 0, False)("vision_tower.x", info) == (128, 0)


def test_tp_transports_features_and_negative_offset_bit_for_bit(monkeypatch):
    records, arrays = [], []
    rank = [0]
    def share(values, r, device):
        if r == 0:
            records.append(list(values))
            return list(values)
        return records.pop(0)
    def broadcast(value, source):
        if rank[0] == 0:
            arrays.append(value.copy())
        else:
            value[:] = arrays.pop(0)
    torch = SimpleNamespace(bfloat16=np.uint16, int32=np.int32,
                            empty=lambda shape, dtype, device: np.empty(shape, dtype=dtype))
    distributed = SimpleNamespace(broadcast=broadcast)
    torch.distributed = distributed
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed)
    monkeypatch.setitem(sys.modules, "tensorfold.families.qwen3_5.cuda.decode_tp", SimpleNamespace(_share=share))
    features = np.array([[0, 65535], [1, 32768]], dtype=np.uint16)
    positions = np.array([[0, 1, 1, 2], [0, 1, 2, 3], [0, 1, 1, 2]], dtype=np.int32)
    outgoing = EncodedVision((1, 2), features, positions, -1)
    broadcast_encoded(outgoing, 0, "cpu", hidden=2, prompt_length=4)
    rank[0] = 1
    received = broadcast_encoded(None, 1, "cpu", hidden=2, prompt_length=4)
    assert received.rows == (1, 2) and received.rope_delta == -1
    np.testing.assert_array_equal(received.features, features)
    np.testing.assert_array_equal(received.positions, positions)


def _engine():
    from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine

    engine = object.__new__(Qwen27Engine)
    engine.vision = SimpleNamespace(encode=lambda prepared, prompt: prepared)
    engine.context_window, engine.tp, engine.scheduler = 100, 1, None
    engine.w = object()
    engine.draft, engine.max_rows, engine.allow_copy = None, 12, True
    engine.cache = [([1, 2], SimpleNamespace(pos=2), None)]
    return engine


def test_images_never_reuse_or_pollute_text_prefix_cache(monkeypatch):
    calls = []
    fake = SimpleNamespace(prefill=lambda *args, **kw: (calls.append(kw) or SimpleNamespace(pos=3), 4),
                           draft_decode=lambda *args, **kw: None)
    monkeypatch.setitem(sys.modules, "tensorfold.families.qwen3_5.cuda.decode", fake)
    engine, payload = _engine(), object()
    cache = engine.cache.copy()
    result = engine.generate([1, 2, 3], 1, None, lambda tokens: True, vision=payload)
    assert calls[0]["state"] is None and calls[0]["vision"] is payload
    assert result["cached"] == 0 and engine.cache == cache


def test_image_encoding_is_deferred_to_scheduler_worker(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorfold.families.qwen3_5.cuda.decode",
                        SimpleNamespace(prefill=None, draft_decode=None))
    engine, payload, calls = _engine(), object(), []
    engine.vision.encode = lambda *args: pytest.fail("HTTP thread invoked the image tower")
    engine.scheduler = SimpleNamespace(submit=lambda *args, **kw: calls.append((args, kw)) or {})
    engine.generate([1, 2, 3], 1, None, lambda tokens: True, vision=payload)
    assert calls[0][1] == {"stop_eos": True, "vision": payload}


def test_state_clone_preserves_image_offset_without_importing_cuda():
    path = Path(__file__).parents[1] / "src/tensorfold/families/qwen3_5/cuda/decode.py"
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "clone_state")
    function.returns = None
    function.args.args[0].annotation = None
    state_type = type("State", (), {})
    namespace = {"State": state_type}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    state = state_type()
    state.pos, state.limit, state.rope_delta, state.room = 257, 1024, -192, object()
    state.conv, state.rec, state.kv = [1], [2], [3]
    cloned = namespace["clone_state"](state)
    assert (cloned.pos, cloned.rope_delta, cloned.limit, cloned.room) == (257, -192, 1024, state.room)
    assert cloned.conv == state.conv and cloned.conv is not state.conv
    assert cloned.kv is state.kv                   # one attention list: a grow reaches every clone


def test_a_meta_built_tower_matches_a_normally_built_one_in_the_installed_transformers():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    from tensorfold.vision.qwen_cuda import rotary_frequencies

    raw = {"depth": 1, "hidden_size": 32, "num_heads": 2, "intermediate_size": 64, "patch_size": 4,
           "spatial_merge_size": 2, "temporal_patch_size": 2, "in_channels": 3, "out_hidden_size": 32,
           "num_position_embeddings": 16, "deepstack_visual_indexes": []}
    config = Qwen3_5VisionConfig(**raw)
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    built = Qwen3_5VisionModel(config).eval()
    with torch.device("meta"):
        meta = Qwen3_5VisionModel(config)
    meta.load_state_dict(built.state_dict(), strict=True, assign=True)
    rotary_frequencies(meta.rotary_pos_emb, raw, "cpu")
    for name, buffer in built.rotary_pos_emb.named_buffers():
        assert torch.equal(dict(meta.rotary_pos_emb.named_buffers())[name], buffer)
    pixels = torch.randn(16, 3 * 2 * 4 * 4)
    grid = torch.tensor([[1, 4, 4]])
    with torch.inference_mode():
        want = built(pixels, grid_thw=grid, return_dict=True).pooler_output
        got = meta.eval()(pixels, grid_thw=grid, return_dict=True).pooler_output
    assert torch.equal(got, want)


def _quantized_checkpoint(path, *, listed=None, damage=None):
    """A depth-1 tower like Flash Next 7c4f1bc1's: MXFP8 attention, first MLP and merger linears, an NVFP4 second
    MLP linear (fp32 weight_scale_2), floating-point norms, biases and embeddings; real, valid payload bytes."""
    rng = np.random.default_rng(7)
    h, mid, merged = 32, 64, 128
    vision = {"model_type": "qwen3_5_vision", "hidden_size": h, "out_hidden_size": h, "depth": 1, "patch_size": 2,
              "temporal_patch_size": 2, "spatial_merge_size": 2, "in_channels": 3, "intermediate_size": mid,
              "num_heads": 2, "num_position_embeddings": 4}
    linears = {"blocks.0.attn.qkv": ([3 * h, h], "mxfp8"), "blocks.0.attn.proj": ([h, h], "mxfp8"),
               "blocks.0.mlp.linear_fc1": ([mid, h], "mxfp8"), "blocks.0.mlp.linear_fc2": ([h, mid], "nvfp4"),
               "merger.linear_fc1": ([merged, merged], "mxfp8"), "merger.linear_fc2": ([h, merged], "mxfp8")}
    layers = {f"model.visual.{name}": {"quant_algo": kind.upper(), "group_size": 32 if kind == "mxfp8" else 16}
              for name, (_, kind) in linears.items()}
    layers.update(listed or {})
    config = {"model_type": "qwen4_exp", "vision_config": vision, "text_config": {
        "hidden_size": h, "head_dim": 8, "rope_parameters": {"mrope_interleaved": True,
        "mrope_section": [2, 1, 1], "partial_rotary_factor": 1}},
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                                "quantized_layers": layers}}
    (path / "config.json").write_text(json.dumps(config))

    def e4m3(shape):                     # e4m3 bytes without the NaN codes 0x7f / 0xff
        b = rng.integers(0, 256, size=shape, dtype=np.uint8)
        return np.where((b & 0x7F) == 0x7F, b - 1, b).astype(np.uint8)

    data = {"patch_embed.proj.weight": ("BF16", rng.standard_normal([h, 2, 2, 2, 3]))}
    for name, shape in {"patch_embed.proj.bias": [h], "pos_embed.weight": [4, h], "merger.norm.weight": [h],
                        "merger.norm.bias": [h], "blocks.0.norm1.weight": [h], "blocks.0.norm1.bias": [h],
                        "blocks.0.norm2.weight": [h], "blocks.0.norm2.bias": [h]}.items():
        data[name] = ("BF16", rng.standard_normal(shape))
    for name, ((n, k), kind) in linears.items():
        data[f"{name}.bias"] = ("BF16", rng.standard_normal([n]))
        if kind == "mxfp8":
            data[f"{name}.weight"] = ("F8_E4M3", e4m3([n, k]))
            data[f"{name}.weight_scale"] = ("U8", rng.integers(118, 128, size=[n, k // 32], dtype=np.uint8))
        else:
            data[f"{name}.weight"] = ("U8", rng.integers(0, 256, size=[n, k // 2], dtype=np.uint8))
            data[f"{name}.weight_scale"] = ("F8_E4M3", rng.integers(0x20, 0x60, size=[n, k // 16], dtype=np.uint8))
            data[f"{name}.weight_scale_2"] = ("F32", np.array(0.0123, dtype=np.float32).reshape([]))
            data[f"{name}.input_scale"] = ("F32", np.array(0.5, dtype=np.float32).reshape([]))
    if damage:
        damage(data)
    entries, blobs, offset = {}, [], 0
    for name, (dtype, array) in data.items():
        if dtype == "BF16":
            raw = (np.asarray(array, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()
        else:
            raw = np.ascontiguousarray(array).tobytes()
        entries["model.visual." + name] = {"dtype": dtype, "shape": list(np.shape(array)),
                                           "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    head = json.dumps(entries).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(head)) + head + b"".join(blobs))
    return entries, data, linears


def test_quantized_tower_headers_count_its_bf16_weights(tmp_path):
    from tensorfold.vision.qwen_cuda import vision_layout

    entries, _, linears = _quantized_checkpoint(tmp_path)
    config, resident, schemes = vision_layout(tmp_path)
    assert config["model_type"] == "qwen3_5_vision"
    assert schemes == {name: kind for name, (_, kind) in linears.items()}
    floats = sum(int(np.prod(e["shape"])) * 2 for name, e in entries.items()
                 if not name.rsplit(".", 1)[0].removeprefix("model.visual.") in linears)
    weights = sum(n * k * 2 for (n, k), _ in linears.values())
    assert resident == floats + weights + sum(n * 2 for (n, _), _ in linears.values())   # + the bf16 biases
    transform = weight_transform(lambda *args: (0, 0), True, 0)
    assert sum(transform(name, e)[0] for name, e in entries.items()) == resident
    assert checkpoint_vision(tmp_path) == (config, resident)


@pytest.mark.parametrize("algo", ["NVFP4", "W4A16_NVFP4", "w4a16_nvfp4"])
def test_weight_only_nvfp4_listings_load_as_nvfp4(tmp_path, algo):
    """7c4f1bc1 lists its NVFP4 fc2 linears as W4A16_NVFP4 (weight-only, no input_scale; s4-gate refused it): the same
    stored weights as an NVFP4 listing, so both are accepted; a listing names the scheme, not the activations."""
    from tensorfold.vision.qwen_cuda import vision_layout

    fc2 = "blocks.0.mlp.linear_fc2"
    _quantized_checkpoint(tmp_path, listed={f"model.visual.{fc2}": {"quant_algo": algo, "group_size": 16}})
    assert vision_layout(tmp_path)[2][fc2] == "nvfp4"


def test_a_weight_only_nvfp4_listing_on_mxfp8_weights_refuses(tmp_path):
    from tensorfold.vision.qwen_cuda import vision_layout

    _quantized_checkpoint(tmp_path, listed={"model.visual.blocks.0.attn.qkv": {"quant_algo": "W4A16_NVFP4", "group_size": 16}})
    with pytest.raises(ValueError, match="config lists W4A16_NVFP4"):
        vision_layout(tmp_path)


@pytest.mark.parametrize("damage", ["no_scale_2", "scale_shape", "listed", "stray_scale", "nan_dtype"])
def test_incomplete_or_mislabelled_quantized_towers_refuse_before_loading(tmp_path, damage):
    fc2 = "blocks.0.mlp.linear_fc2"
    listed = {f"model.visual.{fc2}": {"quant_algo": "MXFP8", "group_size": 32}} if damage == "listed" else None
    edit = {"no_scale_2": lambda d: d.pop(f"{fc2}.weight_scale_2"),
            "scale_shape": lambda d: d.update({f"{fc2}.weight_scale": ("F8_E4M3", np.zeros([32, 3], np.uint8))}),
            "listed": None,
            "stray_scale": lambda d: d.update({"blocks.0.norm1.weight_scale": ("U8", np.zeros([1], np.uint8))}),
            "nan_dtype": lambda d: d.update({f"{fc2}.weight_scale_2": ("F16", np.zeros([], np.float16))})}[damage]
    _quantized_checkpoint(tmp_path, listed=listed, damage=edit)
    with pytest.raises(ValueError):
        checkpoint_vision(tmp_path)


def test_quantized_weights_decode_exactly_as_the_shared_reference_decoder(tmp_path):
    torch = pytest.importorskip("torch")
    from tensorfold.cuda.nvfp4.format import dequant
    from tensorfold.vision.qwen_cuda import dequantized

    _, data, linears = _quantized_checkpoint(tmp_path)
    for name, (_, kind) in linears.items():
        parts = {part.rsplit(".", 1)[1]: array for part, (_, array) in data.items() if part.startswith(name + ".")
                 and part.rsplit(".", 1)[1] != "bias"}
        tensors = {p: torch.from_numpy(np.ascontiguousarray(a)) for p, a in parts.items()}
        if kind == "mxfp8":
            tensors["weight"] = tensors["weight"].view(torch.float8_e4m3fn)
        else:
            tensors["weight_scale"] = tensors["weight_scale"].view(torch.float8_e4m3fn)
        weight, alpha = dequantized(kind, tensors, "cpu")
        want = dequant(kind, parts["weight"], parts["weight_scale"], 1.0)      # nvfp4: codes x e4m3, no global
        assert weight.dtype == torch.bfloat16
        assert torch.equal(weight.float(), torch.from_numpy(want))
        assert alpha == (float(parts["weight_scale_2"]) if kind == "nvfp4" else 1.0)


@pytest.mark.parametrize("kind", ["mxfp8", "nvfp4"])
def test_nan_scales_refuse_at_load(kind):
    torch = pytest.importorskip("torch")
    from tensorfold.vision.qwen_cuda import dequantized

    if kind == "mxfp8":
        parts = {"weight": torch.zeros([4, 32], dtype=torch.uint8).view(torch.float8_e4m3fn),
                 "weight_scale": torch.full([4, 1], 255, dtype=torch.uint8)}
    else:
        parts = {"weight": torch.full([4, 8], 0x11, dtype=torch.uint8),
                 "weight_scale": torch.full([4, 1], 0x7F, dtype=torch.uint8).view(torch.float8_e4m3fn),
                 "weight_scale_2": torch.tensor(0.5)}
    with pytest.raises(ValueError):
        dequantized(kind, parts, "cpu")


def test_scaled_linear_scales_the_products_not_the_bias():
    torch = pytest.importorskip("torch")
    from tensorfold.vision.qwen_cuda import _scaled_linear

    g = torch.Generator().manual_seed(3)
    x, w, b = (torch.randn(s, generator=g, dtype=torch.float64) for s in ([5, 7, 16], [12, 16], [12]))
    layer = _scaled_linear()(torch.nn.Parameter(w), torch.nn.Parameter(b), 0.0123)
    got = layer(x)
    assert got.shape == (5, 7, 12)
    assert torch.allclose(got, 0.0123 * (x @ w.T) + b, rtol=1e-12, atol=1e-12)
