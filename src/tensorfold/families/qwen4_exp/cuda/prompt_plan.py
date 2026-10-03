"""Reuse expert weights in wider idle prompt pieces only when the admitted window leaves enough room.

AGX Thor (sm_110) pieces wider than the released 2048 rows (``TENSORFOLD_PREFILL_ROWS``, which the engine reads
with ``geometry.indexed_prefill_rows``) keep a pool guard: ``keeps_pool``.
"""

import os

from tensorfold.cuda.geometry import PREFILL_ROWS, indexed_prompt_bytes

IDLE_ROWS = 4096
POOL_ENV = "TENSORFOLD_THOR_POOL"      # =0: wide pieces keep today's memory gate (no kept ends freed for the pool)
THOR = (11, 0)


def choose(receipt: dict, text: dict, capability: tuple[int, int], *, world: int = 1,
           vision: bool = False, fp8: bool = False) -> tuple[int, int]:
    """Keep the baseline window; spend spare budget on the measured GB10 MLX shape's prompt workspace."""

    quant = text.get("_quantization") or {}
    shape = tuple(text.get(k) for k in ("hidden_size", "num_experts", "num_experts_per_tok",
                                       "moe_intermediate_size", "hc_count", "hc_lowrank"))
    if (capability != (12, 1) or world != 1 or vision or fp8 or shape != (2560, 512, 10, 640, 4, 320)
            or (quant.get("bits"), quant.get("group_size"), quant.get("mode")) != (4, 32, "affine")):
        return PREFILL_ROWS, 0
    extra = indexed_prompt_bytes(text, IDLE_ROWS - PREFILL_ROWS)
    serving = receipt["serving_peak_bytes_estimate"] + extra
    total = max(receipt["startup_peak_bytes_estimate"], serving)
    if total > receipt["budget_bytes"]:
        return PREFILL_ROWS, 0
    workspace = indexed_prompt_bytes(text, IDLE_ROWS)
    receipt.update(prefill_rows=IDLE_ROWS, prompt_workspace_bytes_estimate=workspace,
                   cache_workspace_bytes_estimate=receipt["cache_workspace_bytes_estimate"] + extra,
                   serving_peak_bytes_estimate=serving, total_bytes_estimate=total,
                   full_mapped_working_set_bytes_estimate=total + receipt["mapped_table_bytes"])
    return IDLE_ROWS, workspace


def keeps_pool(rows: int, capability: tuple[int, int]) -> bool:
    """AGX Thor with pieces wider than the released ones: a cache grow first frees kept prompt ends while the GPU
    driver's free pages can't hold it and the reserve, so the extra piece bytes never push a grow into reclaiming
    the mapped n-gram tables (ROWS-2: 7-109 s stalls); released pieces, other devices and ``TENSORFOLD_THOR_POOL=0``
    keep today's gate."""

    off = os.environ.get(POOL_ENV, "1").strip().lower() in ("0", "off", "false", "no")
    return tuple(capability) == THOR and rows > PREFILL_ROWS and not off


def pass_limit(rows: int, live: bool, share: float, round_s: float | None, row_s: float | None,
               minimum: int = 128) -> int:
    """Live replies retain the released pass bound; an idle decoder can use its full prompt workspace."""

    limit = min(rows, PREFILL_ROWS) if live else rows
    if not live or share <= 0 or not round_s or not row_s:
        return limit
    chosen = int(round_s / (share * row_s)) // 64 * 64
    return min(limit, max(minimum, chosen))
