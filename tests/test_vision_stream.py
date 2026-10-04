"""THOR-VISION-STREAM: a one-video request's tower calls run while its frames decode, with the eager path's bits and
its errors at the same points; the scheduler gets the prompt only with every call run."""

import base64
import io
import threading
import time

import numpy as np
import pytest

from tensorfold.server.errors import RequestError
from tensorfold.server.prompts import prepare_images
from tensorfold.vision import qwen_cuda, videos
from tensorfold.vision.images_http import ImageInputError
from tensorfold.vision.qwen_processing import QwenImageProcessor, StreamedVisionPrompt
from tests.test_vision_video import NormalisingProcessor, VideoTokenizer
from tests.test_vision_qwen_mlx import CONFIG

torch = pytest.importorskip("torch")


def _clip(frames=24, rate=8):
    av = pytest.importorskip("av")
    raw = io.BytesIO()
    with av.open(raw, mode="w", format="mp4") as out:
        stream = out.add_stream("mpeg4", rate=rate)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        rng = np.random.default_rng(7)
        for _ in range(frames):
            out.mux(stream.encode(av.VideoFrame.from_ndarray(rng.integers(0, 256, (64, 96, 3), dtype=np.uint8),
                                                             format="rgb24")))
        out.mux(stream.encode())
    return raw.getvalue()


class Front:
    """The CUDA frontend's request-side surface with a fake tower call that returns its pixels (the bits to check)."""
    allow_urls, videos, device = False, True, "cpu"

    def __init__(self, stream: bool, call_s: float = 0.0):
        self.frontend = QwenImageProcessor(CONFIG, NormalisingProcessor(), VideoTokenizer())
        self.stream, self.call_s, self.calls = stream, call_s, []
        self._tower_lock = threading.Lock()

    def video_size(self, frames, height, width):
        return 32, 64

    def prepare(self, *args, **kwargs):
        return self.frontend.prepare(*args, **kwargs)

    def speculate(self, render, **options):
        return qwen_cuda.VideoSpeculation(self, render, options) if self.stream else None

    _video_steps = qwen_cuda.QwenCudaVision._video_steps

    def _video_call(self, pixels, n, h, w):
        self.calls.append((n, h, w, threading.current_thread().name))
        time.sleep(self.call_s)
        return torch.from_numpy(np.array(pixels))


def _prepare(front, data, monkeypatch):
    monkeypatch.setattr(qwen_cuda, "MAX_PATCHES", 16)        # 8 patches a frame group: two groups a tower call
    url = "data:video/mp4;base64," + base64.b64encode(data).decode()
    messages = [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": url}}]}]
    return prepare_images(front, messages, lambda template: "<start><video><end>").vision


def test_streamed_prompt_has_the_eager_bits_and_every_call_run(monkeypatch):
    data = _clip()
    eager, streamed = _prepare(Front(False), data, monkeypatch), _prepare(front := Front(True), data, monkeypatch)
    assert type(eager) is not StreamedVisionPrompt and isinstance(streamed, StreamedVisionPrompt)
    for field in ("token_ids", "rope_delta", "image_spans", "video_spans", "video_hashes"):
        assert getattr(streamed, field) == getattr(eager, field), field
    assert streamed.position_ids.tobytes() == eager.position_ids.tobytes()
    assert streamed.video_grid_thw.tolist() == eager.video_grid_thw.tolist() == [[3, 2, 4]]
    steps = list(qwen_cuda.QwenCudaVision._video_steps(None, eager.video_grid_thw))
    assert sorted(streamed.video_features) == [key for key, *_ in steps] == [(0, 0), (0, 2)]
    for key, done, n, h, w in steps:                         # each call's input: the eager patches' rows, bit for bit
        rows = eager.video_pixel_values[done:done + n * h * w]
        assert streamed.video_features[key].numpy().tobytes() == rows.tobytes()
        assert streamed.video_pixel_values[done:done + n * h * w].tobytes() == rows.tobytes()
    assert {name for *_, name in front.calls} == {"tensorfold-video-early"} and len(front.calls) == 2


def test_the_scheduler_never_waits_on_the_feed(monkeypatch):
    streamed = _prepare(Front(True), _clip(), monkeypatch)
    rows = streamed.video_pixel_values.shape[0]
    assert streamed.video_pixel_values[0:rows].shape == (rows, 1536)     # the decoded frames, not the (closed) feed


@pytest.mark.parametrize("cut", [0.5, 0.9])
def test_a_damaged_video_fails_as_before(monkeypatch, cut):
    data = _clip()
    bad = data[:int(len(data) * cut)]
    errors = []
    for stream in (False, True):
        with pytest.raises(RequestError) as info:
            _prepare(Front(stream), bad, monkeypatch)
        errors.append((type(info.value), str(info.value)))
    assert errors[0] == errors[1]


def test_a_decode_error_is_raised_at_once_not_after_the_early_calls(monkeypatch):
    """The decode fails after 4 frames while a (slow) early tower call runs: the request fails then, as before."""
    real = videos.decode_video

    def failing(data, size, limits, deadline, feed=None):
        class Stop(Exception):
            pass

        def frame(slot, item, _frame=feed.frame):
            _frame(slot, item)
            if slot == 3:
                raise Stop

        feed.frame = frame
        try:
            return real(data, size, limits, deadline, feed)
        except Stop:
            raise ImageInputError("video bytes are invalid or unsupported (Stop); use MP4 or WebM") from None

    monkeypatch.setattr(videos, "decode_video", failing)
    front = Front(True, call_s=2.0)
    t0 = time.monotonic()
    with pytest.raises(RequestError, match=r"invalid or unsupported \(Stop\)"):
        _prepare(front, _clip(), monkeypatch)
    assert time.monotonic() - t0 < 1.5                       # not held for the 2 s early call in flight


def test_a_short_video_prepares_from_the_frames_it_has(monkeypatch):
    """A stream that ends before its planned frames: the eager preparation of what decoded, as before."""
    real = videos.decode_video

    def short(data, size, limits, deadline, feed=None):
        video = real(data, size, limits, deadline, feed)
        return videos.VideoInput(video.frames[:4].copy(), video.indices[:4], video.fps)

    monkeypatch.setattr(videos, "decode_video", short)
    data = _clip()
    eager, streamed = _prepare(Front(False), data, monkeypatch), _prepare(Front(True), data, monkeypatch)
    assert type(streamed) is not StreamedVisionPrompt
    assert streamed.token_ids == eager.token_ids and streamed.video_grid_thw.tolist() == [[2, 2, 4]]
    assert streamed.video_pixel_values.tobytes() == eager.video_pixel_values.tobytes()


def test_streaming_runs_on_thor_only_and_switches_off(monkeypatch):
    from tensorfold.vision import thor_vision

    monkeypatch.setattr(thor_vision, "available", lambda device=None: True)
    assert thor_vision.stream_on()
    monkeypatch.setenv("TENSORFOLD_THOR_VISION_STREAM", "0")
    assert not thor_vision.stream_on()
    monkeypatch.delenv("TENSORFOLD_THOR_VISION_STREAM")
    monkeypatch.setattr(thor_vision, "available", lambda device=None: False)
    assert not thor_vision.stream_on()
