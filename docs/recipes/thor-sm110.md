# Jetson AGX Thor (sm_110)

The Qwen3.8 Flash Next CUDA family runs on Thor behind a capability check (`sm_110`); every path in this
page is inert on other GPUs. Thor-specific kernels route MoE GEMMs through FlashInfer's W4A16 grouped
kernel and DeltaNet's prompt chains through FlashInfer's chunked kernel; sparse-attention prompt scoring,
the shared-expert elementwise tail, hyper-connection fusion and the n-gram draft table's host-side gather
are TensorFold's own Thor kernels. Each has a `TENSORFOLD_THOR_*=0` switch back to the portable path; none
changes accepted output, only which device code produces it (same contract as every other CUDA family:
drafted output equals serial output, a resumed prompt equals that prompt served fresh).

## Release recipe (FR-OPS-17 par-1-fix)

This is the frozen recipe gated for `qwen3.8-flash-next` on one Thor, serving its full native window with
vision on:

```
IMAGE=tensorfold/thor:dev
ENGINE_COMMIT=538ea92            # thor/par-1-fix, on top of c3d3c40 (r53)
TENSORFOLD_PREFILL_ROWS=4096
TENSORFOLD_MEMORY_RESERVE_GIB=15.6
ARGS="--parallel 8 --context 262144 --vision --vision-max-images 50 --vision-image-tokens 16384"
```

Three fixes land in this recipe, each gated separately before being folded in:

- **DEC-2 hang.** `attn_ptrs()`'s per-stream cache key sampled one layer's live pointer; under 8 concurrent
  streams the allocator could reuse that address for a different stream between calls, so the key matched
  by coincidence and a decode round spun forever on stale data. The key is now `State.version`
  (`src/tensorfold/families/qwen4_exp/cuda/thor_dec2.py`), which changes only on a real cache rebuild.
- **Pool-guard false eviction.** The Thor pool guard compared a live prompt's room against
  `driver_pool()` sampled right after `empty_cache()`, which is structurally near-zero once the 26.8 GiB
  n-gram table is mapped — it does not count host `MemAvailable` minus the mapped table, so it evicted kept
  prefixes it did not need to. `_table_aware_pool()` (`src/tensorfold/families/qwen4_exp/cuda/multi.py`)
  fixes the comparison; an 80k-token prompt-reuse chain that cached 2% of its prompt before this fix caches
  100% after it, in a dedicated session.
- **Admission control.** `NoRoom` is now a `CapacityError` (HTTP 503, `Retry-After`, a machine-readable
  `reason`), `admit()` reserves a stream's worst-case extent before it touches a slot
  (`MemoryGate.promise`/`unpromise`/`admits`), and `GET /v1/capacity` reports live/kept streams, free cache
  bytes and tokens, the largest admissible prompt right now and an `overloaded` flag. See `docs/api.md`,
  "Capacity and admission", for the field contract a router can build against. No request is ever killed
  mid-decode for room; a request that cannot fit is refused at admission instead.

## What "native 262144" measures to

At `--parallel 8`, the mapped n-gram table stays fully host-resident only while every live stream's window
stays under roughly 70–72k tokens (the server logs the exact figure for the loaded checkpoint at startup);
past that, table lookups page from disk and prompts slow down. This is a real, measured trade-off of
running 8 full-window streams on one Thor, not a bug: a single request can still grow to the full
262144-token window and complete end to end — measured once at 250144 target tokens, 481 s time-to-first-
token at `--parallel 8` (520 tok/s effective), against 108 s at `--parallel 4` and 346 s at `--parallel 5`
for the same target, all three runs completing without error. 128k-token prefill stays fast at every
`--parallel` tried (4, 5, 8): 2.5–2.7k tok/s. Admission control and `/v1/capacity` are what make the
native-262144 choice safe: a client that reads capacity first and backs off on 429/503 never needs a
window smaller than the checkpoint's own limit.

Kept-prefix cache retention across several concurrently growing long sessions is a separate, open
limitation, not yet fixed: a growing stream's admission can evict another stream's *kept* (idle) prefix,
oldest-eligible first, to make cache room, protecting only the stream currently growing
(`MultiDecoder._evict_kept`, `src/tensorfold/families/qwen4_exp/cuda/multi.py`). A dedicated session with
nothing else resident caches a 33k- or 80k-token conversation at 99–100% on every follow-up turn; several
such sessions growing in round-robin at once do not, measured at worst-case ~1% follow-up cache on 8
sessions of mixed length. No request is ever refused, killed mid-stream or hung by this — it is a
cache-efficiency gap, not a safety one — but it means "N long sessions stay cached at once" is currently a
measured, bounded number well under `--parallel`, not a guarantee of the full count.
