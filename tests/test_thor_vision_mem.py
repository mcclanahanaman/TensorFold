"""THOR-VISION-MEM (Jetson Thor sm_110 vision memory): the workspace split, the host budget, the hold of a prepared
prompt, the arrays dropped after its encode and the trimmer. Deterministic and CPU-only, except the bf16 check, which
also runs on a CUDA device when there is one."""

from __future__ import annotations

import gc
import threading
import time

import numpy as np
import pytest

from tensorfold.vision import thor_vision_mem as tvm
from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the decoder imports)
from tensorfold.vision.qwen_processing import PreparedVisionPrompt, StreamedVisionPrompt

MIB = 1 << 20


def prepared(rows: int = 8, width: int = 1536, video_rows: int = 0):
    return PreparedVisionPrompt((1, 2, 3), np.ones((rows, width), np.float32), np.array([[1, 2, 2]]),
                                np.zeros((3, 1, 3), np.int32), 0, ((0, 1),), ("h",),
                                np.ones((video_rows, width), np.float32) if video_rows else None)


def test_split_default_and_override(monkeypatch):
    monkeypatch.delenv(tvm.HOST_ENV, raising=False)
    assert tvm.split(1024 * MIB, 600 * MIB) == (600 * MIB, 424 * MIB)
    assert tvm.split(512 * MIB, 600 * MIB) == (512 * MIB, 0)          # the GPU bound never exceeds the workspace
    monkeypatch.setenv(tvm.HOST_ENV, "256")
    assert tvm.split(1024 * MIB, 600 * MIB) == (768 * MIB, 256 * MIB)
    monkeypatch.setenv(tvm.HOST_ENV, "2048")
    with pytest.raises(ValueError, match=tvm.HOST_ENV):
        tvm.split(1024 * MIB, 600 * MIB)
    monkeypatch.setenv(tvm.HOST_ENV, "-1")
    with pytest.raises(ValueError, match=tvm.HOST_ENV):
        tvm.split(1024 * MIB, 600 * MIB)


def test_budget_caps_a_big_request_at_the_whole_budget():
    b = tvm.HostBudget(100)
    h = b.hold(250, timeout=0)
    assert h.nbytes == 100 and b.used == 100 and b.peak == 100
    with pytest.raises(TimeoutError):
        b.hold(1, timeout=0)
    assert b.timeouts == 1
    h.release()
    h.release()                                      # idempotent
    assert b.used == 0
    b.hold(1, timeout=0).release()
    assert b.state()["holds_total"] == 2 and b.state()["host_held_bytes"] == 0


def test_budget_waiter_gets_room_when_released():
    b = tvm.HostBudget(100)
    first = b.hold(80, timeout=0)
    got = []

    def wait():
        got.append(b.hold(50, timeout=5))

    t = threading.Thread(target=wait)
    t.start()
    time.sleep(0.05)
    assert not got and b.state()["waiting"] == 1
    first.release()
    t.join(5)
    assert got and got[0].nbytes == 50 and b.used == 50 and b.waits == 1


def test_budget_is_first_come_first_served():
    """A small request arriving after a big waiting one does not overtake it (no starvation of big requests)."""
    b = tvm.HostBudget(100)
    first = b.hold(60, timeout=0)
    order = []

    def take(n, tag):
        h = b.hold(n, timeout=5)
        order.append(tag)
        time.sleep(0.02)
        h.release()

    big = threading.Thread(target=take, args=(100, "big"))
    big.start()
    time.sleep(0.05)
    small = threading.Thread(target=take, args=(10, "small"))          # would fit now (60 + 10), but big is first
    small.start()
    time.sleep(0.05)
    assert order == []
    first.release()
    big.join(5)
    small.join(5)
    assert order == ["big", "small"]


def test_release_arrays_drops_patches_and_features():
    p = prepared(rows=8, width=16, video_rows=4)
    freed = tvm.release_arrays(p)
    assert freed == 8 * 16 * 4 + 4 * 16 * 4
    assert p.pixel_values.shape == (0, 16) and p.pixel_values.dtype == np.float32
    assert p.video_pixel_values.shape == (0, 16)
    base = prepared(rows=2, width=4)
    s = StreamedVisionPrompt(**{f: getattr(base, f) for f in PreparedVisionPrompt.__dataclass_fields__},
                             video_features={(0, 0): object()})
    tvm.release_arrays(s)
    assert s.video_features is None and s.pixel_values.shape == (0, 4)


def test_memory_done_releases_the_hold_and_the_arrays():
    m = tvm.VisionMemory(host_bytes=100, gpu_bytes=50)
    p = prepared(rows=2, width=4)
    m.attach(p, m.hold(70, timeout=0))
    assert m.budget.used == 70
    m.done(p)
    assert m.budget.used == 0 and p.pixel_values.shape == (0, 4) and m.released_bytes == 2 * 4 * 4
    m.done(p)                                        # twice: nothing more
    assert m.budget.used == 0
    st = m.state()
    assert st["gpu_workspace_bytes"] == 50 and st["host_budget_bytes"] == 100


def test_memory_releases_the_hold_when_the_prompt_is_collected():
    """A request that ends before its encode (refused, cancelled, failed) still gives its bytes back."""
    m = tvm.VisionMemory(host_bytes=100, gpu_bytes=0)
    p = prepared()
    m.attach(p, m.hold(100, timeout=0))
    assert m.budget.used == 100
    del p
    gc.collect()
    assert m.budget.used == 0


def test_request_bytes_bounds():
    w = 1536
    img = tvm.request_bytes(images=10, videos=0, image_tokens=16384, video_tokens=0, max_total_pixels=32 << 20,
                            patch_width=w)
    assert img == 3 * (32 << 20) + 2 * 65536 * w + 4 * 16384 * w * 3
    vid = tvm.request_bytes(images=0, videos=1, image_tokens=0, video_tokens=16384, max_total_pixels=0,
                            patch_width=w, frame_bytes=100 << 20, streamed=True)
    assert vid == 2 * (100 << 20) + 4 * 16384 * w
    assert tvm.request_bytes(images=0, videos=0, image_tokens=0, video_tokens=0, max_total_pixels=0,
                             patch_width=w) == 0
    less = tvm.request_bytes(images=1, videos=0, image_tokens=1024, video_tokens=0, max_total_pixels=1 << 20,
                             patch_width=w)
    assert less < img


def test_a_video_off_the_streamed_path_counts_its_whole_patches_and_their_copy():
    """GLM review SHOULD-FIX 1: two videos, a video beside images, or THOR-VISION-STREAM off: ``prepare`` makes every
    video's float32 patches whole (4 bytes a frame byte) and concatenates them, parts and copy alive together."""
    w, frames = 1536, 100 << 20
    kw = dict(images=0, videos=2, image_tokens=0, video_tokens=16384, max_total_pixels=0, patch_width=w,
              frame_bytes=frames)
    assert tvm.request_bytes(**kw) == 2 * frames + 2 * 4 * frames
    assert tvm.request_bytes(**kw, streamed=True) == 2 * frames + 4 * 16384 * w < tvm.request_bytes(**kw)


def _vision(stream: bool):
    from tensorfold.vision.qwen_cuda import QwenCudaVision

    v = QwenCudaVision.__new__(QwenCudaVision)
    v.config = {"in_channels": 3, "temporal_patch_size": 2, "patch_size": 16}
    v.stream_videos = stream
    return v


@pytest.mark.parametrize("images,videos,stream,streamed", [(0, 1, True, True), (0, 1, False, False),
                                                          (0, 2, True, False), (1, 1, True, False)])
def test_host_bytes_takes_the_streamed_bound_only_for_one_video_alone(monkeypatch, images, videos, stream, streamed):
    """The request's bound follows the path ``prepare_images`` takes: THOR-VISION-STREAM for one video and no image."""
    from types import SimpleNamespace

    monkeypatch.delenv("TENSORFOLD_VIDEO_TOKENS", raising=False)
    limits = SimpleNamespace(max_visual_tokens=16384, max_total_pixels=32 << 20)
    frames = 4 * 16384 * 2 * 16 * 16 * 3 * videos
    want = tvm.request_bytes(images=images, videos=videos, image_tokens=16384, video_tokens=16384,
                             max_total_pixels=32 << 20, patch_width=1536, frame_bytes=frames, streamed=streamed)
    assert _vision(stream).host_bytes(images=images, videos=videos, limits=limits) == want


@pytest.mark.parametrize("video_tokens", [1024, 4096, 16384])
def test_a_kept_hold_covers_its_continuations_encode(video_tokens):
    """GLM review LOW 2: a gated request's continuation encodes again under the request's kept hold. By then its
    decoded pixels, frame copies and processing temporaries are gone; what stays (its patches: bf16 image rows, a
    non-streamed video's float32 array, a streamed video's frames) plus that encode's host copy of one tower call
    (bf16 rows; a streamed video's calls all ran early) fits the bound. A second hold from the budget is not needed,
    and would wait forever on a kept hold that is the whole budget."""
    from tensorfold.vision.qwen_cuda import MAX_PATCHES

    w = 1536
    patches = 4 * 16384
    img = tvm.request_bytes(images=10, videos=0, image_tokens=16384, video_tokens=0, max_total_pixels=32 << 20,
                            patch_width=w)
    assert img >= 2 * patches * w + min(MAX_PATCHES, patches) * 2 * w
    for n in (1, 2):
        frames = 4 * video_tokens * w * n                          # T x p x p x 3 RGB bytes a patch (w), 4 a token
        whole = tvm.request_bytes(images=0, videos=n, image_tokens=0, video_tokens=video_tokens, max_total_pixels=0,
                                  patch_width=w, frame_bytes=frames)
        assert whole >= 4 * frames + min(MAX_PATCHES, 4 * video_tokens * n) * 2 * w
    frames = 4 * video_tokens * w
    early = tvm.request_bytes(images=0, videos=1, image_tokens=0, video_tokens=video_tokens, max_total_pixels=0,
                              patch_width=w, frame_bytes=frames, streamed=True)
    assert early >= frames


def test_the_pool_leaves_out_the_measured_serving_memory_the_budget_unchanged():
    """THOR-KV-FIT's design point leaves at least ``HOST_FLOOR_BYTES`` out of the pool for serving vision (the
    release's 1 GiB workspace splits 640 MiB tower + 384 MiB host); what requests may hold at once stays the part."""
    assert tvm.HOST_FLOOR_BYTES == 704 * MIB
    gpu, host = tvm.split(1024 * MIB, tvm.GPU_BYTES)
    assert (gpu, host) == (640 * MIB, 384 * MIB) and tvm.pool_host(host) == 704 * MIB
    assert tvm.pool_host(2048 * MIB) == 2048 * MIB and tvm.pool_host(0) == 704 * MIB
    assert tvm.VisionMemory(host_bytes=host, gpu_bytes=gpu).budget.total == 384 * MIB


def test_trimmer_never_blocks_and_counts():
    t = tvm.Trimmer()
    t0 = time.perf_counter()
    t.poke()
    t.poke()
    assert time.perf_counter() - t0 < 0.1
    if t._trim is None:
        pytest.skip("no glibc malloc_trim here")
    for _ in range(100):
        if t.trims:
            break
        time.sleep(0.01)
    assert t.trims >= 1


def test_bf16_on_the_host_is_the_bits_torch_tensor_makes():
    """THOR-VISION-MEM converts each image's float32 patches to bf16 on the host when they are made; the tower call
    used to convert them in ``torch.tensor(float32 array, dtype=bf16, device=...)``: the same bits."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(7)
    x = (rng.standard_normal((257, 1536)) * 3).astype(np.float32)
    x[0, :8] = [0.0, -0.0, 1e-40, -1e-40, 3.3895314e38, 1.0 + 2**-8, 1.0 + 3 * 2**-8, -(1.0 + 2**-8)]  # ties, tiny
    host = torch.from_numpy(x).to(torch.bfloat16)
    ref = torch.tensor(x, dtype=torch.bfloat16)
    assert torch.equal(host.view(torch.int16), ref.view(torch.int16))
    if torch.cuda.is_available():
        dev = torch.tensor(x, dtype=torch.bfloat16, device="cuda").cpu()
        assert torch.equal(host.view(torch.int16), dev.view(torch.int16))
        assert torch.equal(host.to("cuda").cpu().view(torch.int16), dev.view(torch.int16))


# --- the processor's bf16 parts (ImageRows) and the server's hold ----------------------------------------------------

from tests.test_vision_qwen_mlx import CONFIG, ImageProcessor, Tokenizer, image  # noqa: E402
from tests.test_vision_server import Frontend, image_messages  # noqa: E402


class RandomPatches(ImageProcessor):
    def __call__(self, *, images, **kwargs):
        self.calls.append((images, kwargs))
        rng = np.random.default_rng(len(self.calls))
        return {"pixel_values": (rng.standard_normal((16, 1536)) * 2).astype(np.float32),
                "image_grid_thw": np.array([[1, 4, 4]])}


def test_image_rows_hold_each_images_patches_as_the_towers_bf16_bits():
    torch = pytest.importorskip("torch")
    from tensorfold.vision.qwen_cuda import _bf16_rows
    from tensorfold.vision.qwen_processing import ImageRows, QwenImageProcessor

    prompt = "<start><image><end>" * 3
    images = [image("a"), image("b"), image("c")]
    ref = QwenImageProcessor(CONFIG, RandomPatches(), Tokenizer()).prepare(prompt, images, max_visual_tokens=48)
    got = QwenImageProcessor(CONFIG, RandomPatches(), Tokenizer()).prepare(prompt, images, max_visual_tokens=48,
                                                                          image_rows=_bf16_rows)
    assert isinstance(got.pixel_values, ImageRows) and got.pixel_values.shape == ref.pixel_values.shape == (48, 1536)
    assert got.pixel_values.nbytes == ref.pixel_values.nbytes // 2
    for a, b in ((0, 48), (0, 16), (16, 48), (5, 37), (32, 48), (47, 48)):
        want = torch.tensor(ref.pixel_values[a:b], dtype=torch.bfloat16)        # what the tower call used to make
        assert torch.equal(got.pixel_values[a:b].view(torch.int16), want.view(torch.int16))
    assert got.token_ids == ref.token_ids and np.array_equal(got.position_ids, ref.position_ids)
    assert got.image_spans == ref.image_spans and np.array_equal(got.image_grid_thw, ref.image_grid_thw)
    numpy_rows = QwenImageProcessor(CONFIG, RandomPatches(), Tokenizer()).prepare(prompt, images, max_visual_tokens=48,
                                                                                 image_rows=lambda a: a)
    assert np.array_equal(numpy_rows.pixel_values[3:40], ref.pixel_values[3:40])
    with pytest.raises(TypeError):
        got.pixel_values[3]
    with pytest.raises(ValueError):
        got.pixel_values[0:10:2]


class MemFrontend(Frontend):
    def __init__(self, host=100, need=60, fail=False):
        super().__init__()
        self.memory = tvm.VisionMemory(host_bytes=host, gpu_bytes=0)
        self.need, self.fail = need, fail

    def host_bytes(self, *, images, videos, limits):
        self.counts = (images, videos)
        return self.need

    def prepare(self, rendered, images, *, max_prompt_tokens, **budget):
        if self.fail:
            raise ValueError("expanded image prompt exceeds the context limit")
        return prepared()


def test_prepare_images_holds_the_bound_until_the_encode():
    from tensorfold.server.prompts import prepare_images

    front = MemFrontend(host=100, need=60)
    rendered = prepare_images(front, image_messages(), str)
    assert front.counts == (1, 0) and front.memory.budget.used == 60
    front.memory.done(rendered.vision)                # the engine's encode ran
    assert front.memory.budget.used == 0 and rendered.vision.pixel_values.shape[0] == 0


def test_a_failed_or_dropped_request_gives_its_bytes_back():
    from tensorfold.server.prompts import prepare_images

    front = MemFrontend(host=100, need=60, fail=True)
    with pytest.raises(Exception, match="exceeds"):
        prepare_images(front, image_messages(), str)
    assert front.memory.budget.used == 0
    front.fail = False
    rendered = prepare_images(front, image_messages(), str)
    assert front.memory.budget.used == 60
    del rendered                                       # refused or cancelled before its encode
    gc.collect()
    assert front.memory.budget.used == 0


def test_a_busy_budget_refuses_after_the_image_wait(monkeypatch):
    from tensorfold.server import prompts
    from tensorfold.server.errors import CapacityError

    monkeypatch.setattr(prompts, "IMAGE_WAIT_S", 0.05)
    front = MemFrontend(host=100, need=60)
    first = prompts.prepare_images(front, image_messages(), str)
    with pytest.raises(CapacityError):
        prompts.prepare_images(front, image_messages(), str)
    assert front.memory.budget.timeouts == 1 and front.memory.budget.used == 60
    front.memory.done(first.vision)
    second = prompts.prepare_images(front, image_messages(), str)
    assert front.memory.budget.used == 60 and second.vision is not first.vision


@pytest.mark.parametrize("threads", [0, 3])
def test_frames_scaled_in_place_keep_the_stacked_pixels(monkeypatch, threads):
    """THOR-VISION-MEM: a video's sampled frames scaled straight into one array are the frames np.stack made, and a
    streamed feed sees the same frames."""
    av = pytest.importorskip("av")
    import io

    from tensorfold.vision import videos
    from tensorfold.vision.videos import VideoLimits

    raw = io.BytesIO()
    with av.open(raw, mode="w", format="mp4") as out:
        stream = out.add_stream("mpeg4", rate=8)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        rng = np.random.default_rng(5)
        for _ in range(24):
            frame = av.VideoFrame.from_ndarray(rng.integers(0, 256, (64, 96, 3), dtype=np.uint8), format="rgb24")
            out.mux(stream.encode(frame))
        out.mux(stream.encode())
    limits = VideoLimits(fps=4.0)
    monkeypatch.setattr(videos, "SCALE_THREADS", threads)
    clips, feeds = [], []
    for in_place in (False, True):
        monkeypatch.setattr(videos, "FRAMES_IN_PLACE", in_place)
        feed = videos.FrameFeed()
        clips.append(videos.decode_video(raw.getvalue(), lambda n, h, w: (32, 64), limits, float("inf"), feed))
        feeds.append(feed.take(0, len(clips[-1].indices)))
    assert clips[0].indices == clips[1].indices and len(clips[0].indices) == 12
    assert np.array_equal(clips[0].frames, clips[1].frames) and clips[0].content_hash == clips[1].content_hash
    assert np.array_equal(feeds[0], feeds[1]) and np.array_equal(feeds[1], clips[1].frames)


def test_the_workspace_host_part_stays_out_of_the_pool_the_room_unchanged(allocations, monkeypatch):  # noqa: F811
    """THOR-VISION-MEM: with the workspace split (GPU part G in the pool, host part H out of it), the pool is H smaller
    and the caches' room is the room the whole workspace in the pool gave."""
    import importlib
    from types import SimpleNamespace

    from tests.test_cuda_growing_caches import decoder, weights
    from tests.test_thor_kv_pool import mods

    multi, state, torch = mods()
    thor_kv = importlib.import_module("tensorfold.families.qwen4_exp.cuda.thor_kv")
    GIB = 1 << 30
    rooms = []
    for gpu, host in ((GIB, 0), (GIB // 2 + GIB // 8, GIB // 2 - GIB // 8)):
        dec = decoder(multi, state, torch, room=0)
        dec.capacity, dec.keep, dec.window_bytes, dec.vision_bytes = 65536, 8, 10 * GIB, gpu
        dec.vision_host = host
        dec._ref_state = state.State(weights(), 256, 4, "bf16", limit=65536)
        dec.graphs = SimpleNamespace(budget=GIB, demand=True)
        dec.memory_gate.live = lambda: 12 * GIB
        asked = []
        monkeypatch.setattr(thor_kv, "reserve", lambda size, asked=asked: asked.append(size) or size)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        note = dec.reserve_pool()
        assert asked == [11 * GIB - host]
        rooms.append(dec.memory_gate.room)
        assert ("host buffers (THOR-VISION-MEM)" in note) == bool(host)
    assert rooms[0] == rooms[1]


def test_a_kept_prompt_survives_its_encode_and_its_continuation_releases_only_its_copy():
    """A gated request (thinking budget, required tool call) encodes its images again in its continuation: the first
    encode keeps the arrays and the hold; a continued copy's encode drops only the copy's references."""
    from tensorfold.vision.qwen_processing import continued

    m = tvm.VisionMemory(host_bytes=100, gpu_bytes=0)
    p = prepared(rows=4, width=8)
    m.attach(p, m.hold(60, timeout=0))
    m.keep(p)
    m.done(p)
    assert p.pixel_values.shape == (4, 8) and m.budget.used == 60
    cfg = {"image_token_id": 99, "vision_config": {"spatial_merge_size": 2}}
    try:
        q = continued(p, p.token_ids + (7,), cfg)
    except Exception:                                   # a fixture prompt without real spans: copy the fields
        q = PreparedVisionPrompt(**{f: getattr(p, f) for f in PreparedVisionPrompt.__dataclass_fields__})
    m.done(q)
    assert q.pixel_values.shape == (0, 8) and p.pixel_values.shape == (4, 8) and m.budget.used == 60
    del p
    gc.collect()
    assert m.budget.used == 0


@pytest.mark.parametrize("calls", [1, 2])
def test_encode_returns_the_placeholder_rows_and_feeds_the_tower_the_same_bits(monkeypatch, calls):
    """The image loop's pixel slice must not shadow the features' rows (vfix-f1b: every image request failed with
    "cannot access local variable 'rows'"); float32 patches and THOR-VISION-MEM's bf16 ImageRows give the tower the
    same bf16 bits, in one tower call or one an image."""
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from tensorfold.vision import qwen_cuda
    from tensorfold.vision.qwen_cuda import QwenCudaVision, _bf16_rows
    from tensorfold.vision.qwen_processing import ImageRows

    merge, hidden, width = 2, 8, 3 * 2 * 16 * 16
    monkeypatch.setattr(qwen_cuda, "MAX_PATCHES", 8 if calls == 2 else qwen_cuda.MAX_PATCHES)
    prompt = (1, 9, 2, 9, 9, 3)                         # image 1: one token (2 x 2 patches), image 2: two (2 x 4)
    grid = np.array([[1, 2, 2], [1, 2, 4]])
    pixels = np.random.default_rng(0).standard_normal((12, width), dtype=np.float32)
    positions = np.tile(np.arange(len(prompt), dtype=np.int64), (3, 1, 1))

    def run(values):
        seen = []

        def tower(x, grid_thw, return_dict, **_):
            seen.append(x.clone())
            n = int(x.shape[0]) // merge**2
            return SimpleNamespace(pooler_output=x[::merge**2, :hidden].float() + torch.arange(n)[:, None])

        vision = QwenCudaVision.__new__(QwenCudaVision)
        vision.config = {"spatial_merge_size": merge, "out_hidden_size": hidden, "in_channels": 3,
                         "temporal_patch_size": 2, "patch_size": 16}
        vision.videos, vision.media_tokens, vision.device, vision._thor_attn, vision.tower = False, 9, "cpu", False, tower
        ready = PreparedVisionPrompt(prompt, values, grid, positions, 0, ((1, 2), (3, 5)), ("a", "b"))
        return vision._encode(ready, prompt), seen

    plain, seen_plain = run(pixels)
    rows, seen_rows = run(ImageRows([_bf16_rows(pixels[:4]), _bf16_rows(pixels[4:])]))
    assert plain.rows == rows.rows == (1, 3, 4) and tuple(plain.features.shape) == (3, hidden)
    assert len(seen_plain) == len(seen_rows) == calls
    for a, b in zip(seen_plain, seen_rows):
        assert a.dtype == b.dtype == torch.bfloat16 and torch.equal(a.view(torch.int16), b.view(torch.int16))
    assert torch.equal(plain.features.view(torch.int16), rows.features.view(torch.int16))


def test_a_closed_feed_drops_its_frames_and_leaves_no_cycle():
    """A plan points back at its feed: once the request closes the feed, its frames (views of the video's one array)
    and the plan go with plain reference counting, never waiting for a full garbage collection [vmem-a1: 45-93 MiB a
    video until the trim's gc.collect]; a reader after the close gets FeedClosed."""
    import weakref

    from tensorfold.vision import videos

    was = gc.isenabled()
    gc.disable()
    try:
        into = np.zeros((4, 8, 16, 3), np.uint8)
        feed = videos.FrameFeed()
        feed.begin(4, 8, 16, range(4), 2.0)
        for slot in range(4):
            feed.frame(slot, into[slot])
        assert feed.take(0, 4).shape == (4, 8, 16, 3) and feed.plan.feed is feed
        frames, alive = weakref.ref(into), weakref.ref(feed)
        feed.close()
        assert feed.plan is None and feed.frames == []
        with pytest.raises(videos.FeedClosed):
            feed.take(0, 1)
        feed.frame(1, into[1])                           # a scale finishing after the close is dropped
        del into
        assert frames() is None
        del feed
        assert alive() is None
    finally:
        if was:
            gc.enable()


def test_a_failed_speculation_keeps_no_patches_or_frames_alive_once_dropped(monkeypatch):
    """A video speculation whose feed closes before its next call's frames arrive keeps its error without the
    traceback (whose frame held the speculation, so the call's patches stayed until a full collection): with the
    collector off, the call's patches, the frames and the speculation all go when the request drops it."""
    pytest.importorskip("torch")
    import weakref

    from tensorfold.vision import qwen_cuda, videos
    from tests.test_vision_stream import Front

    class Recording(Front):
        refs: list = []

        def _video_call(self, pixels, n, h, w):
            self.refs.append(weakref.ref(pixels))
            return super()._video_call(pixels, n, h, w)

    monkeypatch.setattr(qwen_cuda, "MAX_PATCHES", 16)    # 8 patches a frame group: two groups a tower call
    was = gc.isenabled()
    gc.disable()
    try:
        front = Recording(True)
        spec = front.speculate(lambda: "<start><video><end>")
        into = np.random.default_rng(3).integers(0, 256, (8, 32, 64, 3), dtype=np.uint8)
        spec.feed.begin(8, 32, 64, range(8), 8.0)
        for slot in range(4):                            # the first call's two groups; the second call waits
            spec.feed.frame(slot, into[slot])
        deadline = time.monotonic() + 30
        while not front.refs and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(front.refs) == 1
        spec.cancel()
        spec.thread.join(30)
        assert isinstance(spec.error, videos.FeedClosed) and spec.error.__traceback__ is None and spec.prepared is None
        refs = [front.refs[0], weakref.ref(into), weakref.ref(spec), weakref.ref(spec.feed)]
        del into, spec
        assert [r() is None for r in refs] == [True] * 4
    finally:
        if was:
            gc.enable()
