"""Flash Next video input on CUDA: frame groups ride the image path, each its own timestamped vision block."""

import numpy as np
import pytest

from tensorfold.vision.images import ImageInputError, ImageSource, split_images
from tensorfold.vision.qwen_processing import QwenImageProcessor, media_positions
from tensorfold.vision.videos import (DEFAULT_VIDEO_LIMITS, VideoInput, VideoLimits, VideoSource, sample_indices,
                                      video_source)
from tests.test_vision_qwen_mlx import CONFIG, ImageProcessor, Tokenizer, image

DATA = "data:video/mp4;base64,AAAA"


class VideoTokenizer(Tokenizer):
    def convert_ids_to_tokens(self, token):
        return {11: "<video>"}.get(token) or super().convert_ids_to_tokens(token)

    def convert_tokens_to_ids(self, token):
        return {"<video>": 11}.get(token) or super().convert_tokens_to_ids(token)

    def __call__(self, text, **kwargs):
        encode = super().__call__
        return {"input_ids": [t for chunk in text.split("<video>")
                             for t in encode(chunk, **kwargs)["input_ids"] + [11]][:-1]}


def video(frames=4, side=32):
    return VideoInput(np.zeros((frames, side, side, 3), np.uint8), tuple(range(frames)), 2.0)


def test_each_frame_group_is_its_own_block_and_text_resumes_past_it():
    ids = [100, 8, 11, 9, 101, 8, 11, 9, 102]
    pos, delta, spans, frames = media_positions(ids, np.zeros((0, 3)), [[2, 2, 2]], CONFIG)
    assert spans == () and frames == ((2, 3), (6, 7))
    assert pos[:, 0].tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8]] * 3
    assert delta == 0


def test_video_tokens_without_a_video_are_refused():
    with pytest.raises(ValueError, match="Video inputs are not supported"):
        media_positions([8, 11, 9], np.zeros((0, 3)), None, CONFIG)
    with pytest.raises(ValueError, match="without a corresponding video"):
        media_positions([8, 11, 9, 8, 11, 9], np.zeros((0, 3)), [[1, 2, 2]], CONFIG)


def test_video_parts_split_only_where_videos_are_on_and_do_not_count_as_images():
    parts = [{"type": "video_url", "video_url": {"url": DATA}}, {"type": "text", "text": "what moves?"}]
    with pytest.raises(ImageInputError, match="audio and video are unsupported"):
        split_images([{"role": "user", "content": parts}])
    template, sources = split_images([{"role": "user", "content": parts}], allow_videos=True)
    assert template[0]["content"][0] == {"type": "video"} and isinstance(sources[0], VideoSource)
    many = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}] * 4 + parts
    _, sources = split_images([{"role": "user", "content": many}], allow_videos=True)
    assert sum(isinstance(s, ImageSource) for s in sources) == 4                 # the default image count, not 5
    with pytest.raises(ImageInputError, match="at most 2 videos"):
        split_images([{"role": "user", "content": parts * 3}], allow_videos=True)


@pytest.mark.parametrize("url, allow, match", [
    ("http://example.com/a.mp4", True, "data URLs or public HTTPS"),
    ("https://example.com/a.mp4", False, "video URLs are off"),
    ("data:video/mp4;base64," + "A" * (DEFAULT_VIDEO_LIMITS.max_encoded_bytes * 2), True, "encoded byte limit"),
])
def test_video_urls_follow_the_image_rules(url, allow, match):
    with pytest.raises(ImageInputError, match=match):
        video_source({"url": url}, DEFAULT_VIDEO_LIMITS, allow)
    assert video_source({"url": DATA}, DEFAULT_VIDEO_LIMITS, False).url == DATA


def test_a_video_fits_the_request_body_limit():
    assert DEFAULT_VIDEO_LIMITS.max_total_encoded_bytes * 4 // 3 < 32 * 1024**2


def test_frames_are_sampled_at_two_a_second_across_the_whole_video():
    limits = VideoLimits()
    assert sample_indices(300, 30.0, limits).tolist() == np.linspace(0, 299, 20).round().astype(int).tolist()
    assert len(sample_indices(10, 30.0, limits)) == limits.min_frames
    assert len(sample_indices(30 * 3600, 30.0, limits)) == limits.max_frames


def test_each_frame_group_is_timed_at_its_frames_mean():
    clip = VideoInput(np.zeros((3, 32, 32, 3), np.uint8), (0, 2, 4), 2.0)
    assert clip.timestamps(2) == [0.5, 2.0]                   # an odd last frame repeats


def test_prepare_expands_a_video_into_timestamped_frame_groups():
    front = QwenImageProcessor(CONFIG, ImageProcessor(), VideoTokenizer())
    prepared = front.prepare("a<start><video><end>b", [], videos=[video()])
    assert prepared.video_grid_thw.tolist() == [[2, 2, 2]]
    assert prepared.video_pixel_values.shape == (8, 1536)
    assert len(prepared.video_spans) == 2 and prepared.image_spans == ()
    assert prepared.visual_tokens == 2 and prepared.video_hashes == (video().content_hash,)
    text = "".join(chr(t) if t > 20 else f"<{t}>" for t in prepared.token_ids)
    assert "<0.2 seconds><8><11><9>" in text and "<1.2 seconds><8><11><9>" in text


def test_prepare_mixes_images_and_videos_in_prompt_order():
    front = QwenImageProcessor(CONFIG, ImageProcessor(), VideoTokenizer())
    prepared = front.prepare("<start><image><end><start><video><end>", [image()], videos=[video()])
    first_video = prepared.video_spans[0][0]
    assert prepared.image_spans[0][1] <= first_video
    assert prepared.visual_tokens == 4 + 2                    # a 4 x 4 image merged 2 x 2, two frame groups


class NormalisingProcessor(ImageProcessor):
    image_mean, image_std = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
    rescale_factor = 1 / 255


@pytest.mark.parametrize("frames", [4, 7])
@pytest.mark.parametrize("threads", [1, 3, 8])
def test_fused_video_patches_keep_the_processors_bits(frames, threads):
    """THOR-VISION-PREP: the fused, threaded patches equal the processor's bit for bit (an odd last frame repeats)."""
    rng = np.random.default_rng(10 * frames + threads)
    clip = VideoInput(rng.integers(0, 256, (frames, 64, 96, 3), dtype=np.uint8), tuple(range(frames)), 2.0)
    front = QwenImageProcessor(CONFIG, NormalisingProcessor(), VideoTokenizer())
    want, grid = front._video_patches(clip)
    front.video_threads = threads
    got, fused_grid = front._video_patches(clip)
    assert got.dtype == want.dtype == np.float32 and got.shape == want.shape == (grid.prod(), 1536)
    assert got.tobytes() == want.tobytes() and fused_grid.tolist() == grid.tolist()


def test_fused_video_patches_run_on_thor_only_and_switch_off(monkeypatch):
    from tensorfold.vision import thor_vision

    monkeypatch.setattr(thor_vision, "available", lambda device=None: True)
    assert thor_vision.prep_threads() >= 1
    monkeypatch.setenv("TENSORFOLD_THOR_VISION_PREP", "0")
    assert thor_vision.prep_threads() == 0
    monkeypatch.delenv("TENSORFOLD_THOR_VISION_PREP")
    monkeypatch.setattr(thor_vision, "available", lambda device=None: False)
    assert thor_vision.prep_threads() == 0


def test_frames_scaled_on_a_pool_keep_the_serial_pixels(monkeypatch):
    """THOR-VISION-PREP: the sampled frames' scaling on a pool gives the serial loop's frames, in order."""
    av = pytest.importorskip("av")
    import io

    from tensorfold.vision import videos

    raw = io.BytesIO()
    with av.open(raw, mode="w", format="mp4") as out:
        stream = out.add_stream("mpeg4", rate=8)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        rng = np.random.default_rng(3)
        for i in range(24):
            frame = av.VideoFrame.from_ndarray(rng.integers(0, 256, (64, 96, 3), dtype=np.uint8), format="rgb24")
            out.mux(stream.encode(frame))
        out.mux(stream.encode())
    limits = VideoLimits(fps=4.0)
    clips = []
    for threads in (0, 3):
        monkeypatch.setattr(videos, "SCALE_THREADS", threads)
        clips.append(videos.decode_video(raw.getvalue(), lambda n, h, w: (32, 64), limits, float("inf")))
    assert clips[0].indices == clips[1].indices and len(clips[0].indices) == 12
    assert clips[0].frames.shape == (12, 32, 64, 3) and clips[0].content_hash == clips[1].content_hash
