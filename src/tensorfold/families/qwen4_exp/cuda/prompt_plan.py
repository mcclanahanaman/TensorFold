"""Reuse expert weights in wider idle prompt pieces only when the admitted window leaves enough room.

Two rules spend spare budget on wider idle pieces: the measured GB10 MLX shape takes 4096 rows; on AGX Thor (sm_110)
``TENSORFOLD_PREFILL_ROWS`` sets them (a multiple of 64 from 2048 to 16384; unset: the released 2048), counting the
Thor path's own buffers that grow with a piece (its experts' scratch). Live replies keep the released bound
(``pass_limit``), and a budget that cannot hold the wider pieces at the admitted window keeps the released ones.
"""

import os

from tensorfold.cuda.geometry import PREFILL_ROWS, indexed_prompt_bytes

IDLE_ROWS = 4096
ROWS_ENV = "TENSORFOLD_PREFILL_ROWS"   # AGX Thor: an idle prompt piece's rows (unset or empty: PREFILL_ROWS)
ROWS_GRID = 64                        # Thor's DeltaNet chunk and expert route tile: pieces on this grid keep every bit
MAX_ROWS = 16384                      # the widest piece checked on Thor; every buffer's element offsets stay in 32 bits
THOR = (11, 0)


def requested() -> int:
    """The idle piece rows ``TENSORFOLD_PREFILL_ROWS`` asks for: a multiple of 64 from PREFILL_ROWS to MAX_ROWS, or
    PREFILL_ROWS when unset or empty; any other value is refused at startup."""

    raw = os.environ.get(ROWS_ENV, "").strip()
    if not raw:
        return PREFILL_ROWS
    rows = int(raw) if raw.isdigit() else -1
    if rows % ROWS_GRID or not PREFILL_ROWS <= rows <= MAX_ROWS:
        raise ValueError(f"{ROWS_ENV}={raw!r}: an idle prompt piece's rows, a multiple of {ROWS_GRID} from "
                         f"{PREFILL_ROWS} to {MAX_ROWS}")
    return rows


def choose(receipt: dict, text: dict, capability: tuple[int, int], *, world: int = 1,
           vision: bool = False, fp8: bool = False, thor=None) -> tuple[int, int]:
    """Keep the baseline window; spend spare budget on the measured GB10 MLX shape's prompt workspace, or on AGX Thor
    on the pieces ``TENSORFOLD_PREFILL_ROWS`` asks for. ``thor(text, rows)``: the bytes the Thor NVFP4 path's own
    buffers hold for a piece of ``rows`` rows (None: not that path)."""

    if capability == THOR:
        return _thor(receipt, text, world, vision, fp8, thor)
    quant = text.get("_quantization") or {}
    shape = tuple(text.get(k) for k in ("hidden_size", "num_experts", "num_experts_per_tok",
                                       "moe_intermediate_size", "hc_count", "hc_lowrank"))
    if (capability != (12, 1) or world != 1 or vision or fp8 or shape != (2560, 512, 10, 640, 4, 320)
            or (quant.get("bits"), quant.get("group_size"), quant.get("mode")) != (4, 32, "affine")):
        return PREFILL_ROWS, 0
    return _widen(receipt, text, IDLE_ROWS, indexed_prompt_bytes(text, IDLE_ROWS - PREFILL_ROWS))


def _thor(receipt: dict, text: dict, world: int, vision: bool, fp8: bool, thor) -> tuple[int, int]:
    """AGX Thor: ``TENSORFOLD_PREFILL_ROWS`` pieces on one GPU's NVFP4 path when the budget holds them; otherwise the
    released pieces, with the reason printed (the receipt untouched)."""

    rows = requested()
    if rows == PREFILL_ROWS:
        return PREFILL_ROWS, 0
    why = ("it runs on one GPU" if world != 1 else "image input keeps them" if vision else
           "FP8 prompts keep them" if fp8 else "it applies to the NVFP4 checkpoint" if thor is None else None)
    if why is None:
        extra = indexed_prompt_bytes(text, rows - PREFILL_ROWS) + thor(text, rows) - thor(text, PREFILL_ROWS)
        chosen = _widen(receipt, text, rows, extra)
        if chosen[0] == rows:
            return chosen
        why = f"the admitted window's budget has no room for {extra / 2**30:.2f} GiB more"
    print(f"[tensorfold] {ROWS_ENV}={rows}: {why}; idle prompt pieces stay {PREFILL_ROWS} rows", flush=True)
    return PREFILL_ROWS, 0


def _widen(receipt: dict, text: dict, rows: int, extra: int) -> tuple[int, int]:
    """``rows``-row idle pieces when the budget holds ``extra`` more bytes at the admitted window (the receipt then
    counts them), else the released pieces with the receipt untouched."""

    serving = receipt["serving_peak_bytes_estimate"] + extra
    total = max(receipt["startup_peak_bytes_estimate"], serving)
    if total > receipt["budget_bytes"]:
        return PREFILL_ROWS, 0
    workspace = indexed_prompt_bytes(text, rows)
    receipt.update(prefill_rows=rows, prompt_workspace_bytes_estimate=workspace,
                   cache_workspace_bytes_estimate=receipt["cache_workspace_bytes_estimate"] + extra,
                   serving_peak_bytes_estimate=serving, total_bytes_estimate=total,
                   full_mapped_working_set_bytes_estimate=total + receipt["mapped_table_bytes"])
    return rows, workspace


def pass_limit(rows: int, live: bool, share: float, round_s: float | None, row_s: float | None,
               minimum: int = 128) -> int:
    """Live replies retain the released pass bound; an idle decoder can use its full prompt workspace."""

    limit = min(rows, PREFILL_ROWS) if live else rows
    if not live or share <= 0 or not round_s or not row_s:
        return limit
    chosen = int(round_s / (share * row_s)) // 64 * 64
    return min(limit, max(minimum, chosen))
