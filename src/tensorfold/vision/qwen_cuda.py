"""Qwen image features and request-local positions for the CUDA lane engine."""

from __future__ import annotations

import contextlib
from contextlib import contextmanager
from dataclasses import dataclass
import functools
import json
import threading
import math
import os
from pathlib import Path
from typing import Any

MAX_PATCHES = 16384                  # one tower call's patches: a 4,096-token image's, the scratch reserved
MAX_REQUEST_PATCHES = 16 * MAX_PATCHES   # a request's (--vision-image-tokens 65536), encoded MAX_PATCHES at a time
MAX_VIDEO_PATCHES = 16 * MAX_PATCHES     # a request's video patches (~65k tokens), encoded MAX_PATCHES at a time
TOKENS_PER_IMAGE = 4096              # one image's visual tokens, whatever budget the request's images share
WORKSPACE_BYTES = 4 * 1024**3
# --vision-offload: the tower (about 0.9 GiB) visits the GPU per image, so the budget keeps room for it plus the
# activations of the largest accepted image (measured peak about 1.2 GiB over idle on a 4,096-token image)
OFFLOAD_WORKSPACE_BYTES = int(2.25 * 1024**3)
FLOATS = ("BF16", "F16", "F32")
# a quantized tower linear's tensors beside its weight (ModelOpt names); they are folded in at load, none stays resident
SCALES = ("weight_scale", "weight_scale_2", "input_scale")
# quantized_layers' quant_algo names for each stored scheme; ModelOpt lists weight-only NVFP4 (no input_scale) as
# W4A16_NVFP4, as Flash Next 7c4f1bc1 does for its tower's mlp.linear_fc2: the same stored weights
ALGOS = {"mxfp8": ("MXFP8",), "nvfp4": ("NVFP4", "W4A16_NVFP4"), "fp8": ("FP8",)}
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)   # FP4 values by code


def rotary_frequencies(rotary: Any, config: dict, device: Any) -> None:
    """Fill the frequency buffers a meta-device build leaves empty, as every transformers version computes them."""
    import torch

    dim = int(config["hidden_size"]) // int(config["num_heads"]) // 2
    theta = float((config.get("rope_parameters") or {}).get("rope_theta", 10000.0))
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))
    for name in ("inv_freq", "original_inv_freq"):
        if name in rotary._buffers:
            rotary._buffers[name] = inv_freq.clone()


@dataclass
class EncodedVision:
    rows: tuple[int, ...]
    features: Any
    positions: Any
    rope_delta: int


def vision_config(model_dir: str | Path) -> dict:
    raw = json.loads((Path(model_dir) / "config.json").read_text())
    config = raw.get("vision_config")
    if not isinstance(config, dict) or config.get("model_type") not in ("qwen3_5", "qwen3_5_vision", "qwen4_exp"):
        raise ValueError("CUDA vision requires a Qwen3.5-compatible vision checkpoint")
    if config.get("deepstack_visual_indexes"):
        raise ValueError("CUDA Qwen vision does not support deepstack image features")
    for key in ("hidden_size", "out_hidden_size", "depth", "patch_size", "temporal_patch_size",
                "spatial_merge_size", "in_channels", "intermediate_size", "num_heads", "num_position_embeddings"):
        if not isinstance(config.get(key), int) or isinstance(config[key], bool) or config[key] <= 0:
            raise ValueError(f"invalid vision configuration: {key}")
    if config["hidden_size"] % config["num_heads"] or config["hidden_size"] // config["num_heads"] % 4:
        raise ValueError("vision attention requires head widths divisible by four")
    if math.isqrt(config["num_position_embeddings"]) ** 2 != config["num_position_embeddings"]:
        raise ValueError("vision position embeddings must form a square grid")
    text = raw.get("text_config", raw)
    if config["out_hidden_size"] != text.get("hidden_size"):
        raise ValueError("vision output width differs from the language embedding width")
    from .rotary import frequency_axes

    rope = text.get("rope_parameters") or {}
    if not rope.get("mrope_interleaved", False):
        raise ValueError("CUDA Qwen vision requires interleaved multimodal rotary positions")
    dims = int(int(text["head_dim"]) * float(rope.get("partial_rotary_factor", 0.25)))
    frequency_axes(dims, rope.get("mrope_section", (11, 11, 10)))
    return config


def _vision_sources(model_dir):
    from .qwen_checkpoint import vision_tensors

    override = os.environ.get("TENSORFOLD_VISION_WEIGHTS")
    return vision_tensors(Path(model_dir), weights_path=Path(override) if override else None)


def _resident_bytes(name: str, info: dict) -> int:
    """A tower tensor's bytes once loaded: weights as bf16 (a packed NVFP4 byte holds two), scales folded in."""
    from tensorfold.cuda.capacity import SIZES

    part = name.rsplit(".", 1)[-1]
    if part in SCALES:
        return 0
    count = math.prod(info["shape"])
    return count * 4 if part == "weight" and info["dtype"] == "U8" else count * max(2, SIZES[info["dtype"]])


def _quantized_layers(model_dir) -> dict:
    """config.json's ModelOpt quantized_layers entries for the tower, by local name ({} when it has none)."""
    from tensorfold.cuda.nvfp4.format import config_block
    from .qwen_checkpoint import vision_key

    block = config_block(json.loads((Path(model_dir) / "config.json").read_text())) or {}
    layers = block.get("quantized_layers") or {}
    return {vision_key(k): v for k, v in layers.items() if isinstance(v, dict) and vision_key(k) is not None}


def _schemes(tensors: dict, shapes: dict, listed: dict) -> dict[str, str]:
    """Each quantized tower linear's scheme (mxfp8, nvfp4 or fp8) from its tensors, checked against the configured
    [N, K], the scheme's stored shapes and the config's quantized_layers; floating-point linears are left out."""
    from tensorfold.cuda.nvfp4.format import scheme

    found = {}
    for key, (n, k) in ((key, shape) for key, shape in shapes.items() if key.endswith(".weight") and len(shape) == 2):
        name = key[:-len(".weight")]
        parts = {p: (tensors[f"{name}.{p}"]["dtype"], tensors[f"{name}.{p}"]["shape"])
                 for p in ("weight", *SCALES) if f"{name}.{p}" in tensors}
        if "weight" not in parts:
            continue                                     # the complete-tower check reports it
        kind = scheme(parts)
        algo = (listed.get(name) or {}).get("quant_algo")
        if kind == "bf16":
            if len(parts) > 1 or algo is not None:
                raise ValueError(f"vision linear {name} stores floating-point weights but scales or a quant_algo")
            continue
        scalar = ("F32", None)
        want = {"mxfp8": ({"weight": ("F8_E4M3", [n, k]), "weight_scale": ("U8", [n, k // 32])}, {}, 32),
                "nvfp4": ({"weight": ("U8", [n, k // 2]), "weight_scale": ("F8_E4M3", [n, k // 16]),
                           "weight_scale_2": scalar}, {"input_scale": scalar}, 16),
                "fp8": ({"weight": ("F8_E4M3", [n, k]), "weight_scale": scalar}, {"input_scale": scalar}, 1)}.get(kind)
        if want is None:
            raise ValueError(f"unsupported vision weight storage {kind}: {name}")
        required, optional, group = want
        if k % group or set(required) - set(parts) or set(parts) - set(required) - set(optional):
            raise ValueError(f"vision linear {name} does not hold a complete {kind} weight")
        for part, (dtype, shape) in parts.items():
            expect, size = {**required, **optional}[part]
            if dtype != expect or (shape not in ([], [1]) if size is None else shape != size):
                raise ValueError(f"vision tensor shapes differ from the checkpoint configuration: {name}.{part}")
        if algo is not None and str(algo).upper() not in ALGOS[kind]:
            raise ValueError(f"vision linear {name} stores {kind} weights but the config lists {algo}")
        found[name] = kind
    return found


def vision_layout(model_dir: str | Path) -> tuple[dict, int, dict[str, str]]:
    """Validate vision tensor headers before any model or accelerator allocation: the configuration, the tower's
    resident bytes (bf16 weights) and each quantized linear's scheme (see dequantized)."""
    from tensorfold.cuda.capacity import SIZES
    config = vision_config(model_dir)
    sources = _vision_sources(model_dir)
    tensors = {k: value[1] for k, value in sources.items()}
    for name, (path, info, begin) in sources.items():
        shape, offsets = info.get("shape", ()), info.get("data_offsets", ())
        scalar = name.rsplit(".", 1)[-1] in SCALES and shape == []      # a per-tensor scale
        if (info.get("dtype") not in SIZES or not isinstance(shape, list) or not (shape or scalar)
                or any(type(d) is not int or d <= 0 for d in shape) or len(offsets) != 2
                or any(type(d) is not int for d in offsets) or offsets[0] < 0
                or offsets[1] - offsets[0] != math.prod(shape) * SIZES[info["dtype"]]
                or begin + offsets[1] > path.stat().st_size):
            raise ValueError(f"invalid or unsupported vision tensor range: {name}")
    h, mid, merged, out = (config["hidden_size"], config["intermediate_size"],
                           config["hidden_size"] * config["spatial_merge_size"]**2, config["out_hidden_size"])
    shapes = {"patch_embed.proj.bias": [h], "pos_embed.weight": [config["num_position_embeddings"], h],
              "merger.norm.weight": [h], "merger.norm.bias": [h], "merger.linear_fc1.weight": [merged, merged],
              "merger.linear_fc1.bias": [merged], "merger.linear_fc2.weight": [out, merged],
              "merger.linear_fc2.bias": [out]}
    for layer in range(config["depth"]):
        for part, width, inputs in (("norm1", h, None), ("norm2", h, None), ("attn.qkv", 3 * h, h),
                                    ("attn.proj", h, h), ("mlp.linear_fc1", mid, h), ("mlp.linear_fc2", h, mid)):
            shapes[f"blocks.{layer}.{part}.weight"] = [width] if inputs is None else [width, inputs]
            shapes[f"blocks.{layer}.{part}.bias"] = [width]
    schemes = _schemes(tensors, shapes, _quantized_layers(model_dir))
    scales = {f"{name}.{part}" for name in schemes for part in SCALES if f"{name}.{part}" in tensors}
    if set(tensors) != set(shapes) | {"patch_embed.proj.weight"} | scales:
        raise ValueError("checkpoint needs the complete Qwen vision tower with floating-point, MXFP8, NVFP4 or FP8 "
                         "linears; use its original MLX checkpoint")
    packed = {f"{name}.weight" for name in schemes}
    if any(tensors[key]["shape"] != expected for key, expected in shapes.items() if key not in packed):
        raise ValueError("vision tensor shapes differ from the checkpoint configuration")
    if any(v["dtype"] not in FLOATS for key, v in tensors.items() if key not in packed and key not in scales):
        raise ValueError("CUDA vision requires floating-point vision weights outside its quantized linears")
    h, p, t, channels = (config[k] for k in ("hidden_size", "patch_size", "temporal_patch_size", "in_channels"))
    shape = tensors["patch_embed.proj.weight"]["shape"]
    if shape not in ([h, t, p, p, channels], [h, channels, t, p, p]):
        raise ValueError("unsupported vision patch convolution layout")
    return config, sum(_resident_bytes(key, info) for key, info in tensors.items()), schemes


def checkpoint_vision(model_dir: str | Path) -> tuple[dict, int]:
    """Validate vision tensor headers before any model or accelerator allocation."""
    config, resident, _ = vision_layout(model_dir)
    return config, resident


def dequantized(kind: str, parts: dict, device) -> tuple[Any, float]:
    """A quantized tower linear's bf16 weight and the per-tensor fp32 scale it leaves out (1.0 for MXFP8).

    MXFP8 values (e4m3 times an e8m0 power of two per 32 inputs) and NVFP4's code times e4m3 products (one scale per
    16 inputs) have at most six significant bits, so bf16 holds them exactly; NVFP4's weight_scale_2 and FP8's
    weight_scale stay one scalar the GEMM applies to its fp32 sums (_scaled_linear), so the tower computes with exactly
    the checkpoint's values. Activation scales are unused: activations stay bf16. Raises unless every value is exact."""
    import torch

    weight = parts["weight"].to(device)
    if kind == "nvfp4":
        n, half = weight.shape
        codes = torch.stack([weight & 0xF, weight >> 4], dim=-1).reshape(n, 2 * half)   # low nibble = even input
        table = torch.tensor(E2M1, dtype=torch.float32, device=device)
        values = table[codes.long()] * parts["weight_scale"].to(device).to(torch.float32).repeat_interleave(16, dim=1)
        alpha = float(parts["weight_scale_2"].reshape(-1)[0].float())
    elif kind == "mxfp8":
        bits = parts["weight_scale"].to(device).view(torch.uint8).to(torch.int32)
        # 2**(e - 127) as fp32 bits: the exponent field, or 2**-127's subnormal pattern for e = 0; 255 is NaN
        power = torch.where(bits == 0, 1 << 22, bits << 23).view(torch.float32)
        values = weight.to(torch.float32) * torch.where(bits == 255, math.nan, power).repeat_interleave(32, dim=1)
        alpha = 1.0
    elif kind == "fp8":
        values, alpha = weight.to(torch.float32), float(parts["weight_scale"].reshape(-1)[0].float())
    else:
        raise ValueError(f"unsupported vision weight storage {kind}")
    result = values.to(torch.bfloat16)
    if (not (math.isfinite(alpha) and alpha > 0) or not bool(torch.isfinite(values).all())
            or not torch.equal(result.to(torch.float32), values)):
        raise ValueError(f"a {kind} vision weight does not load exactly as bf16 values and a positive scale")
    return result, alpha


@functools.lru_cache(maxsize=None)
def _scaled_linear():
    import torch

    class ScaledLinear(torch.nn.Module):
        """A tower linear whose stored weight leaves out a per-tensor fp32 scale: alpha * (x @ weight.T) + bias, alpha
        applied to the GEMM's fp32 sums before its one rounding (torch.addmm), as a bf16 linear rounds once."""

        def __init__(self, weight, bias, alpha: float):
            super().__init__()
            self.weight, self.bias, self.alpha = weight, bias, alpha

        def forward(self, x):
            rows = x.reshape(-1, x.shape[-1])
            bias = rows.new_zeros(()) if self.bias is None else self.bias
            return torch.addmm(bias, rows, self.weight.t(), alpha=self.alpha).reshape(*x.shape[:-1], -1)

        def extra_repr(self) -> str:
            return f"in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}, alpha={self.alpha!r}"

    return ScaledLinear


def weight_transform(base, enabled: bool, rank: int, offload: bool = False):
    def transform(name, info):
        from .qwen_checkpoint import vision_key

        if enabled and vision_key(name) is not None:
            if rank != 0 or offload or os.environ.get("TENSORFOLD_VISION_WEIGHTS"):
                return 0, 0                     # offloaded: resident in host RAM between images
            return _resident_bytes(vision_key(name), info), 0
        return base(name, info)
    return transform


def capacity_geometry(base, model_dir, enabled: bool, rank: int, workspace: int = WORKSPACE_BYTES,
                      offload: bool = False):
    def geometry(text):
        from tensorfold.cuda.capacity import Geometry

        result = base(text)
        if not enabled:
            return result
        external_weights = 0
        if rank == 0:
            _, tower_bytes = checkpoint_vision(model_dir)
            if os.environ.get("TENSORFOLD_VISION_WEIGHTS") and not offload:
                external_weights = tower_bytes
        reserve = (OFFLOAD_WORKSPACE_BYTES if offload else workspace) if rank == 0 else 128 * 1024**2
        return Geometry(lambda slots: result.bytes_at(slots) + reserve + external_weights,
                        result.reserve, result.minimum_slots)
    return geometry


def _bf16_rows(pixels):
    """THOR-VISION-MEM: one image's float32 patches as host bf16 rows, the bits ``torch.tensor(pixels,
    dtype=torch.bfloat16, device=...)`` gives the tower (that copy converts on the host, round to nearest even)."""
    import torch

    return torch.from_numpy(pixels).to(torch.bfloat16)


class QwenCudaVision:
    """Only the image tower is loaded; the CUDA family retains all language computation."""

    def __init__(self, model_dir, device, allow_urls: bool = False, offload: bool = False):
        self.allow_urls = allow_urls
        self.offload = offload
        self._lock = threading.Lock()   # one image on the GPU at a time when the tower is offloaded
        resident = "cpu" if offload else device
        import torch
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
        from safetensors import safe_open
        from .qwen_processing import QwenImageProcessor
        from .qwen_checkpoint import vision_key

        self.memory = None              # THOR-VISION-MEM (``thor_vision_mem.VisionMemory``): the engine's, on sm_110
        self.config, self.weight_bytes, schemes = vision_layout(model_dir)
        self.frontend = QwenImageProcessor.from_directory(model_dir)
        raw = json.loads((Path(model_dir) / "config.json").read_text())
        self.image_token = int(raw["image_token_id"])
        # videos: Flash Next's frontend (the frame groups ride the image path; tested on that checkpoint)
        self.videos = raw.get("model_type") == "qwen4_exp" and "video_token_id" in raw
        self.media_tokens = frozenset({self.image_token} | ({int(raw["video_token_id"])} if self.videos else set()))
        self.device = device
        config = Qwen3_5VisionConfig(**{k: v for k, v in self.config.items()
                                      if k not in ("model_type", "deepstack_visual_indexes")})
        config._attn_implementation = "sdpa"
        with torch.device("meta"):
            tower = Qwen3_5VisionModel(config)
        tensors, stored = {}, {}
        by_file = {}
        for key, (path, info, begin) in _vision_sources(model_dir).items():
            by_file.setdefault(path, {})[key] = info
        for path, selected in by_file.items():
            with safe_open(str(path), framework="pt", device="cpu") as source:
                names = {vision_key(name): name for name in source.keys()}
                for key in selected:
                    value = source.get_tensor(names[key])
                    owner, _, part = key.rpartition(".")
                    if owner in schemes and part != "bias":    # a quantized linear's weight and scales: decoded below
                        stored[key] = value
                        continue
                    if key == "patch_embed.proj.weight" and value.shape[-1] == self.config["in_channels"]:
                        value = value.permute(0, 4, 1, 2, 3).contiguous()
                    tensors[key] = value.to(device=resident, dtype=torch.bfloat16)
        alphas = {}
        for name, kind in schemes.items():
            parts = {key.rpartition(".")[2]: value for key, value in stored.items() if key.rpartition(".")[0] == name}
            tensors[f"{name}.weight"], alphas[name] = dequantized(kind, parts, resident)
        tower.load_state_dict(tensors, strict=True, assign=True)
        for name, alpha in alphas.items():
            if alpha != 1.0:                    # the per-tensor scale rides the GEMM (exact weights, one rounding)
                owner, _, leaf = name.rpartition(".")
                linear = getattr(tower.get_submodule(owner), leaf)
                setattr(tower.get_submodule(owner), leaf, _scaled_linear()(linear.weight, linear.bias, alpha))
        if schemes:
            counts = {kind: sum(1 for k in schemes.values() if k == kind) for kind in sorted(set(schemes.values()))}
            print(f"[tensorfold] vision: {', '.join(f'{n} {k.upper()}' for k, n in counts.items())} tower linears "
                  f"loaded exactly (bf16 values; per-tensor scales applied to the GEMM sums)", flush=True)
        rotary_frequencies(tower.rotary_pos_emb, self.config, resident)
        self.tower = tower.eval()
        self._thor_attn = None      # THOR-VISION-ATTN (sm_110 only; an offloaded tower keeps transformers' path)
        if not offload:
            from . import thor_vision

            self._thor_attn = thor_vision.install(self.tower, device)
            if self._thor_attn:
                parts = [p for p, on in (("the rotary embedding in one pass", self._thor_attn["rope"]),
                                         ("one flash call a layer", self._thor_attn["varlen"])) if on]
                print(f"[tensorfold] Thor (sm_110): vision attention with {' and '.join(parts)}, transformers' bits "
                      f"(TENSORFOLD_THOR_VISION=0 keeps transformers' path)", flush=True)
        from . import thor_vision, videos

        # THOR-VISION-STREAM: a one-video request's tower calls run while its frames decode (needs the one-flash-call
        # attention: no SDPA flags, which are process-wide, are set off the scheduler's thread)
        self.stream_videos = bool(self._thor_attn and self._thor_attn.get("varlen") and thor_vision.stream_on(device))
        self._tower_lock = threading.Lock()
        if self.stream_videos:
            print("[tensorfold] Thor (sm_110): a video's tower calls run as its frames decode, the same bits "
                  "(TENSORFOLD_THOR_VISION_STREAM=0 encodes after the decode)", flush=True)
        threads = thor_vision.prep_threads(device)       # THOR-VISION-PREP, CPU work: an offloaded tower too
        if threads:
            self.frontend.video_threads = videos.SCALE_THREADS = threads
            thor_vision.numpy_hugepage_advice_off()
            print(f"[tensorfold] Thor (sm_110): video frames scaled and patched on {threads} threads, numpy's "
                  f"hugepage advice off, the same bits (TENSORFOLD_THOR_VISION_PREP=0 keeps the CPU path)", flush=True)

    def warm(self):
        """Load tower kernels at startup using one small, merge-aligned image grid."""
        import torch

        merge = self.config["spatial_merge_size"]
        patches = merge * merge
        width = self.config["in_channels"] * self.config["temporal_patch_size"] * self.config["patch_size"]**2
        with self._on_gpu(), torch.inference_mode():
            self.tower(torch.zeros((patches, width), dtype=torch.bfloat16, device=self.device),
                       grid_thw=torch.tensor([[1, merge, merge]], device=self.device), return_dict=True)
            torch.cuda.synchronize()

    def prepare(self, *args, **kwargs):
        kwargs.setdefault("max_image_tokens", TOKENS_PER_IMAGE)
        if self.memory is not None:     # THOR-VISION-MEM: each image's patches kept as bf16 rows, the tower's bits
            kwargs.setdefault("image_rows", _bf16_rows)
        return self.frontend.prepare(*args, **kwargs)

    def use_memory(self, memory) -> None:
        """THOR-VISION-MEM (set by the engine on sm_110): requests hold ``memory``'s host budget until their encode,
        their arrays are dropped after it, and a video's sampled frames are scaled into one array (the same pixels)."""
        from . import videos

        self.memory = memory
        videos.FRAMES_IN_PLACE = True

    def host_bytes(self, *, images: int, videos: int, limits) -> int:
        """THOR-VISION-MEM: a request's host bound before its decode (``thor_vision_mem.request_bytes``) from its image
        and video counts and the server's image limits (its visual-token budget and decoded-pixel total)."""
        from . import thor_vision_mem

        width = self.config["in_channels"] * self.config["temporal_patch_size"] * self.config["patch_size"]**2
        video_tokens = int(os.environ.get("TENSORFOLD_VIDEO_TOKENS") or 16384)
        # a video's sampled frames: four patches a token, each patch T frames of p x p RGB pixels
        frame_bytes = 4 * video_tokens * self.config["temporal_patch_size"] * self.config["patch_size"]**2 * 3
        return thor_vision_mem.request_bytes(
            images=images, videos=videos, image_tokens=min(int(limits.max_visual_tokens), MAX_REQUEST_PATCHES // 4),
            video_tokens=video_tokens, max_total_pixels=int(limits.max_total_pixels), patch_width=width,
            frame_bytes=frame_bytes * videos)

    def video_size(self, frames: int, height: int, width: int) -> tuple[int, int]:
        return self.frontend.video_size(frames, height, width)
    @contextmanager
    def _on_gpu(self):
        """The tower on the GPU for the block; when offloaded, one caller at a time, and back to host RAM after."""
        if not self.offload:
            yield
            return
        import torch

        with self._lock:
            try:
                self.tower.to(self.device)
                yield
            finally:
                self.tower.to("cpu")
                torch.cuda.empty_cache()

    def encode(self, prepared, prompt) -> EncodedVision:
        try:
            if getattr(self, "stream_videos", False):  # THOR-VISION-STREAM: one tower call at a time beside early calls
                with self._tower_lock:
                    return self._encode(prepared, prompt)
            with self._on_gpu():
                return self._encode(prepared, prompt)
        finally:
            if self.memory is not None:              # THOR-VISION-MEM: the patches dropped, the hold back, a trim
                self.memory.done(prepared)

    def speculate(self, render, **prepare_options):
        """THOR-VISION-STREAM: a ``VideoSpeculation`` for a one-video request, None when off (``stream_videos``)."""
        return VideoSpeculation(self, render, prepare_options) if getattr(self, "stream_videos", False) else None

    def _video_call(self, pixels, n: int, h: int, w: int):
        """One tower call: ``n`` frame groups of ``h`` x ``w`` patches (float32 rows) -> their bf16 features."""
        import torch

        pixels = torch.tensor(pixels, dtype=torch.bfloat16, device=self.device)
        grids = torch.tensor([[n, h, w]], dtype=torch.int64, device=self.device)
        features = self.tower(pixels, grid_thw=grids, return_dict=True, **self._longest([h * w])).pooler_output
        return features.to(dtype=torch.bfloat16)

    def _video_steps(self, videos):
        """``_encode``'s tower calls over the videos, in order: (key, first patch row, groups, h, w)."""
        done = 0
        for v, (t, h, w) in enumerate(videos):
            t, h, w = int(t), int(h), int(w)
            step = max(1, MAX_PATCHES // (h * w))
            for g in range(0, t, step):
                n = min(step, t - g)
                yield (v, g), done, n, h, w
                done += n * h * w

    def _longest(self, segments) -> dict:
        """THOR-VISION-ATTN's one flash call a layer: the call's longest attention segment from the host."""
        return {"max_seqlen": max(segments)} if self._thor_attn else {}

    def _encode(self, prepared, prompt) -> EncodedVision:
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel

        if tuple(prompt) != tuple(prepared.token_ids):
            raise ValueError("vision preparation belongs to different prompt tokens")
        grid = prepared.image_grid_thw
        videos = getattr(prepared, "video_grid_thw", None)
        if videos is not None and not self.videos:
            raise ValueError("this server's vision frontend encodes images only")
        if len(grid.shape) != 2 or grid.shape[1] != 3 or any(int(t) != 1 for t in grid[:, 0]):
            raise ValueError("CUDA vision accepts images with one temporal grid, not video")
        merge = self.config["spatial_merge_size"]
        every = list(grid) + ([] if videos is None else list(videos))
        if any(int(value) != value or value <= 0 for row in every for value in row) or any(
                int(h) % merge or int(w) % merge for _, h, w in every):
            raise ValueError("image grids must contain positive merge-aligned dimensions")
        sizes = [int(t) * int(h) * int(w) for t, h, w in grid]
        patches = sum(sizes)
        clips = 0 if videos is None else sum(int(t) * int(h) * int(w) for t, h, w in videos)
        if patches > MAX_REQUEST_PATCHES or max(sizes, default=0) > MAX_PATCHES or (patches <= 0 and clips <= 0):
            raise ValueError(f"image request exceeds the CUDA vision budget of {MAX_PATCHES} patches an image and "
                             f"{MAX_REQUEST_PATCHES} a request")
        if clips > MAX_VIDEO_PATCHES:
            raise ValueError(f"video request exceeds the CUDA vision budget of {MAX_VIDEO_PATCHES} patches")
        patch_width = (self.config["in_channels"] * self.config["temporal_patch_size"] * self.config["patch_size"]**2)
        if tuple(prepared.pixel_values.shape) != (patches, patch_width) or (
                videos is not None and tuple(prepared.video_pixel_values.shape) != (clips, patch_width)):
            raise ValueError("image patch tensor has an invalid shape")
        if tuple(prepared.position_ids.shape) != (3, 1, len(prompt)):
            raise ValueError("image positions must have shape (3, 1, prompt tokens)")
        frames = tuple(getattr(prepared, "video_spans", ()))
        spans = sorted(tuple(prepared.image_spans) + frames)
        rows = tuple(i for start, end in spans for i in range(start, end))
        positions = prepared.position_ids[:, 0, :].tolist()
        validate_encoded(rows, positions, prepared.rope_delta, prompt, self.media_tokens,
                         ((patches + clips) // merge**2, self.config["out_hidden_size"]),
                         self.config["out_hidden_size"])
        with torch.inference_mode(), sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            blocks = {}                                  # span start -> its features, in the tower's order
            starts = iter(start for start, _ in prepared.image_spans)
            # images never attend to one another: runs of whole images, at most MAX_PATCHES a tower call (one call,
            # as before, whenever the request fits it), so the scratch stays what one full-size image needs
            for begin, end in image_runs(sizes, MAX_PATCHES):
                done = sum(sizes[:begin])
                # a copy: the prepared arrays are read-only, and a tensor may not share them; THOR-VISION-MEM's bf16
                # rows (converted on the host as this conversion does) are copied as they are
                rows = prepared.pixel_values[done:done + sum(sizes[begin:end])]
                pixels = (rows.to(device=self.device) if isinstance(rows, torch.Tensor) else
                          torch.tensor(rows, dtype=torch.bfloat16, device=self.device))
                del rows
                grids = torch.tensor(grid[begin:end], dtype=torch.int64, device=self.device)
                features = self.tower(pixels, grid_thw=grids, return_dict=True,
                                      **self._longest([int(h) * int(w) for _, h, w in grid[begin:end]])).pooler_output
                for part in features.to(dtype=torch.bfloat16).split([s // merge**2 for s in sizes[begin:end]]):
                    blocks[next(starts)] = part
            if clips:
                # frame groups never attend to one another either: a video encodes a bounded run of them at a time
                spans_left = iter(frames)
                early = getattr(prepared, "video_features", None) or {}     # THOR-VISION-STREAM: calls already run
                for key, done, n, h, w in self._video_steps(videos):
                    features = early.get(key)
                    if features is None:
                        features = self._video_call(prepared.video_pixel_values[done:done + n * h * w], n, h, w)
                    for part in features.split(h * w // merge**2):
                        start, end = next(spans_left)
                        if end - start != part.shape[0]:
                            raise ValueError("video frame features do not match their placeholders")
                        blocks[start] = part
            features = torch.cat([blocks[start] for start, _ in spans]).contiguous()
        if tuple(features.shape) != (len(rows), self.config["out_hidden_size"]):
            raise ValueError("vision tower returned a different number of image features")
        return EncodedVision(rows, features, torch.tensor(positions, dtype=torch.int32, device=self.device),
                             prepared.rope_delta)


class VideoSpeculation:
    """THOR-VISION-STREAM: a one-video request's preparation and tower calls on their own thread while the request's
    thread decodes the video exactly as before (``feed`` hears each frame). It never raises into the request: any
    failure, a decode that ends early or differs from the plan, or ``cancel`` leaves ``take`` returning None, and the
    request prepares and encodes after the decode as it always did (the same errors at the same points). The scheduler
    gets the prompt only after the decode, every call run, so it never waits for a frame."""

    def __init__(self, vision: QwenCudaVision, render, prepare_options: dict) -> None:
        from .videos import FrameFeed

        self.vision, self.render, self.options = vision, render, prepare_options
        self.feed = FrameFeed()
        self.prepared = self.error = None
        self.features: dict = {}
        self.cancelled = False
        self.thread = threading.Thread(target=self._run, name="tensorfold-video-early", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        import torch

        try:
            plan = self.feed.wait_plan()
            prepared = self.vision.prepare(self.render(), [], videos=[plan], **self.options)
            if prepared.image_grid_thw.shape[0] or prepared.video_grid_thw is None:
                raise ValueError("a speculated video prompt holds one video and no image")
            cuda = torch.device(self.vision.device).type == "cuda"
            with torch.cuda.device(self.vision.device) if cuda else contextlib.nullcontext(), torch.inference_mode():
                for key, done, n, h, w in self.vision._video_steps(prepared.video_grid_thw):
                    pixels = prepared.video_pixel_values[done:done + n * h * w]     # waits for these frames
                    if self.cancelled:
                        return
                    with self.vision._tower_lock:
                        self.features[key] = self.vision._video_call(pixels, n, h, w)
                if cuda:
                    torch.cuda.current_stream().synchronize()     # the features are ready for any stream
            self.prepared = prepared
        except Exception as exc:  # noqa: BLE001 - the request's own path reports its errors where it always did
            self.error = exc

    def cancel(self) -> None:
        self.cancelled = True
        self.feed.close()

    def take(self, video):
        """The prompt prepared from ``video`` (the decode's result) with its tower calls run, or None."""
        plan = self.feed.plan
        if plan is None or video.indices != plan.indices or tuple(video.frames.shape) != plan.shape or \
                video.fps != plan.fps:
            self.cancel()
            return None
        self.thread.join()
        if self.prepared is None or self.cancelled:
            return None
        from .qwen_processing import LazyVideoPatches, StreamedVisionPrompt

        lazy = self.prepared.video_pixel_values
        frames = video.frames
        pixels = LazyVideoPatches(lambda a, b: frames[a:b], frames.shape[0], self.prepared.video_grid_thw[0],
                                  lazy._temporal, lazy._make)
        fields = {f: getattr(self.prepared, f) for f in self.prepared.__dataclass_fields__}
        fields.update(video_pixel_values=pixels, video_hashes=(video.content_hash,), video_features=self.features)
        return StreamedVisionPrompt(**fields)


def image_runs(sizes, limit: int) -> list[tuple[int, int]]:
    """Consecutive [begin, end) runs of images whose patches fit ``limit`` together (an image alone always fits)."""
    runs, begin, total = [], 0, 0
    for i, size in enumerate(sizes):
        if i > begin and total + size > limit:
            runs.append((begin, i))
            begin, total = i, 0
        total += size
    if sizes:
        runs.append((begin, len(sizes)))
    return runs


def validate_encoded(rows, positions, delta: int, prompt, image_token, feature_shape, hidden: int) -> None:
    """Reject a payload that could overwrite text rows or misalign the language cache; ``image_token``: the
    placeholder id, or the set of them (images and videos)."""
    n = len(prompt)
    media = image_token if isinstance(image_token, (set, frozenset, tuple)) else {image_token}
    if list(rows) != [i for i, token in enumerate(prompt) if token in media]:
        raise ValueError("vision feature rows must match every image placeholder exactly")
    if not rows or tuple(feature_shape) != (len(rows), hidden):
        raise ValueError("vision feature count or width differs from the image placeholders")
    if len(positions) != 3 or any(len(axis) != n for axis in positions):
        raise ValueError("vision positions must contain three coordinates for every prompt token")
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 or v >= 2**31 - 1
           for axis in positions for v in axis):
        raise ValueError("vision positions must be nonnegative int32 coordinates")
    if not isinstance(delta, int) or isinstance(delta, bool) or max(map(max, positions)) + 1 - n != delta:
        raise ValueError("vision decode offset does not follow its prompt positions")


def broadcast_encoded(payload: EncodedVision | None, rank: int, device, *, hidden: int,
                      prompt_length: int) -> EncodedVision | None:
    """Both ranks receive identical features; the vision tower exists only on rank zero."""
    import torch
    import torch.distributed as dist
    from tensorfold.families.qwen3_5.cuda.decode_tp import _share

    meta = _share(([0] if payload is None else [1, payload.rope_delta, *payload.rows]) if rank == 0 else None,
                  rank, device)
    if meta == [0]:
        return None
    if len(meta) < 3 or meta[0] != 1 or len(meta) - 2 > prompt_length:
        raise ValueError("invalid distributed vision metadata")
    rows = tuple(meta[2:])
    if sorted(set(rows)) != list(rows) or rows[0] < 0 or rows[-1] >= prompt_length:
        raise ValueError("distributed vision row indices are outside the prompt")
    features = payload.features if rank == 0 else torch.empty((len(rows), hidden), dtype=torch.bfloat16, device=device)
    positions = payload.positions if rank == 0 else torch.empty((3, prompt_length), dtype=torch.int32, device=device)
    dist.broadcast(features, 0)
    dist.broadcast(positions, 0)
    return EncodedVision(rows, features, positions, meta[1])


def replace_rows(x, payload: EncodedVision, start: int, end: int):
    import torch

    selected = [(i, row - start) for i, row in enumerate(payload.rows) if start <= row < end]
    if selected:
        source, target = zip(*selected)
        source = torch.tensor(source, dtype=torch.int64, device=x.device)
        target = torch.tensor(target, dtype=torch.int64, device=x.device)
        x.index_copy_(0, target, payload.features.index_select(0, source))
    return x
