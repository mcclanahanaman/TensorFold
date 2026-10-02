"""Spec for tools/thor/serve_digest.py (the startup memory digest of a TensorFold Flash Next server log).

    python -m pytest -q tools/thor/test_serve_digest.py

Standard library and pytest only; no GPU, no TensorFold import.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import serve_digest  # noqa: E402

START = ("[tensorfold] CUDA rank 0 startup estimate 87.86 GiB within 106.84 GiB; native 262144, allocated prompt/reply "
         "window 262144, cache slots 262151")
THOR = [
    "[tensorfold] Thor (sm_110): routed NVFP4 experts on FlashInfer's W4A16 grouped GEMM (TENSORFOLD_THOR_FI_MOE=0 "
    "keeps the portable kernel)",
    "[tensorfold] Thor (sm_110): prompt GEMMs on cuBLAS bf16 (192 MXFP8 exact in bf16, 106 bf16, 102 kept); MXFP8 "
    "weights as bf16 copies, 5.47 GiB (TENSORFOLD_THOR_DENSE=0 keeps the portable kernels, "
    "TENSORFOLD_THOR_DENSE_WEIGHTS=copy|dequant)",
    "[tensorfold] Thor (sm_110): DeltaNet prompt chains on FlashInfer's chunked kernel (float16 q/k/v, fp32 state, "
    "64-row chunks; kept points and prompt pieces on the 64-row grid), ready in 0.1s (TENSORFOLD_THOR_GDN=0 keeps "
    "TensorFold's chain)",
    "[tensorfold] Thor (sm_110): sparse-attention prompt kernels: tensor-core block scores, attention on a bf16 shadow "
    "of each layer's 8-bit cache (0.50 GiB) (TENSORFOLD_THOR_QSA=0 keeps TensorFold's)",
]


def flash(room: float, rows: int, full: float = 4.47, streams: int = 4, window: int = 262144) -> str:
    return (f"[tensorfold] Flash Next on CUDA: 1 to 6 MTP drafts a round, a chain stops before a later draft under "
            f"70%; up to {streams} streams, each growing to {window} prompt/reply tokens while memory lasts ({room} "
            f"GiB free for their caches, {full} GiB for one at the full window), eager; int8 KV cache (fp16 scale per "
            f"32 values); n-gram tables read in 0.7s; 0 decode graphs captured; idle prompt pieces {rows} rows; prompt "
            f"kernels warmed in 1.7s")


SERVING = ("[tensorfold] serving qwen3.8-flash-next at http://127.0.0.1:8010/v1 on CUDA (sampling: temperature 1.0, "
           "top_k 20, top_p 0.95; drafts: on; prompts: bf16 activations; context: 262144; loaded in 48.3s)")
R19 = "\n".join(["[tensorfold] loading qwen3.8-flash-next: Qwen3.8 Flash Next (qwen4_exp) on CUDA", START, *THOR,
                 flash(25.4, 2048), SERVING, "INFO:     127.0.0.1:50000 - \"POST /v1/chat/completions HTTP/1.1\" 200"])
WIDE = "\n".join([START, *THOR, "[tensorfold] 8192-row idle prompt workspace 3.35 GiB; planned peak 91.20 GiB at the "
                  "admitted window", flash(21.9, 8192), SERVING])
REFUSED = "\n".join([START, *THOR, "[tensorfold] TENSORFOLD_PREFILL_ROWS=8192: the admitted window's budget has no "
                     "room for 4.10 GiB more; idle prompt pieces stay 2048 rows", flash(25.4, 2048), SERVING])
STOCK = "\n".join([START.replace("87.86", "82.01"), flash(32.1, 2048), SERVING.replace("48.3", "41.0")])
BROKEN = "\n".join([START, "Traceback (most recent call last):", "  File \"x.py\", line 1, in <module>",
                    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"])
KEYS = ["startup_estimate_gib", "budget_gib", "native_window", "window", "cache_slots", "workspace_rows",
        "workspace_gib", "planned_peak_gib", "refused_rows", "refused_why", "streams", "stream_window", "kv_room_gib",
        "full_window_gib", "pass_rows", "dense_copies_gib", "qsa_shadow_gib", "loaded_s", "warmed_s", "errors",
        "reserve_gib", "usable_kv_gib", "full_window_streams"]


def test_the_r19_startup_lines():
    d = serve_digest.digest(R19)
    assert list(d) == KEYS
    assert d["startup_estimate_gib"] == 87.86 and d["budget_gib"] == 106.84
    assert (d["native_window"], d["window"], d["cache_slots"]) == (262144, 262144, 262151)
    assert d["workspace_rows"] is None and d["workspace_gib"] == 0.0 and d["planned_peak_gib"] == 87.86
    assert d["refused_rows"] is None and d["refused_why"] is None
    assert (d["streams"], d["stream_window"], d["kv_room_gib"], d["full_window_gib"]) == (4, 262144, 25.4, 4.47)
    assert d["pass_rows"] == 2048
    assert d["dense_copies_gib"] == 5.47 and d["qsa_shadow_gib"] == 0.5
    assert d["loaded_s"] == 48.3 and d["warmed_s"] == 1.7 and d["errors"] == 0
    assert d["reserve_gib"] == 2.0 and d["usable_kv_gib"] == pytest.approx(23.4) and d["full_window_streams"] == 5


def test_a_wide_workspace_raises_the_reserve():
    d = serve_digest.digest(WIDE)
    assert d["workspace_rows"] == 8192 and d["workspace_gib"] == 3.35 and d["planned_peak_gib"] == 91.2
    assert d["pass_rows"] == 8192 and d["kv_room_gib"] == 21.9
    assert d["reserve_gib"] == 3.35 and d["usable_kv_gib"] == pytest.approx(18.55)
    assert d["full_window_streams"] == 4


def test_a_refused_request_and_a_stock_server():
    d = serve_digest.digest(REFUSED)
    assert d["refused_rows"] == 8192
    assert d["refused_why"] == "the admitted window's budget has no room for 4.10 GiB more"
    assert d["pass_rows"] == 2048 and d["workspace_rows"] is None and d["reserve_gib"] == 2.0
    s = serve_digest.digest(STOCK)
    assert s["startup_estimate_gib"] == 82.01 and s["kv_room_gib"] == 32.1 and s["loaded_s"] == 41.0
    assert s["dense_copies_gib"] is None and s["qsa_shadow_gib"] is None
    assert s["full_window_streams"] == 6                       # (32.1 - 2.0) / 4.47 = 6.7


def test_errors_and_missing_lines():
    d = serve_digest.digest(BROKEN)
    assert d["errors"] == 2
    assert d["kv_room_gib"] is None and d["usable_kv_gib"] is None and d["full_window_streams"] is None
    assert d["pass_rows"] is None and d["planned_peak_gib"] == 87.86
    e = serve_digest.digest("")
    assert list(e) == KEYS and e["startup_estimate_gib"] is None and e["planned_peak_gib"] is None
    assert e["errors"] == 0 and e["workspace_gib"] == 0.0 and e["reserve_gib"] == 2.0


def test_the_last_start_in_a_log_wins():
    d = serve_digest.digest(R19 + "\n" + WIDE)
    assert d["pass_rows"] == 8192 and d["kv_room_gib"] == 21.9 and d["workspace_rows"] == 8192


def test_cli_writes_the_digest(tmp_path, capsys):
    log = tmp_path / "serve_r8192_1.log"
    log.write_text(WIDE)
    out = tmp_path / "mem_r8192_1.json"
    assert serve_digest.main([str(log), "--json", str(out)]) == 0
    assert json.loads(out.read_text()) == serve_digest.digest(WIDE)
    printed = capsys.readouterr().out
    assert "pass rows 8192" in printed and "full-window streams 4" in printed
    assert serve_digest.main([str(tmp_path / "missing.log")]) == 1
