# Jetson AGX Thor (sm_110)

The Qwen3.8 Flash Next CUDA family runs on Thor behind a capability check (`sm_110`); every path on this page is
inert on other GPUs. Thor-specific kernels route MoE GEMMs through FlashInfer's W4A16 grouped kernel and DeltaNet's
prompt chains through FlashInfer's chunked kernel; sparse-attention prompt scoring, the shared-expert elementwise tail,
hyper-connection fusion and the n-gram table's host-side gather are TensorFold's own Thor paths. Each has a
`TENSORFOLD_THOR_*` switch back to the portable path (listed below); none changes accepted output, only which device
code produces it (the contract every CUDA family keeps: drafted output equals serial output, a resumed prompt equals
that prompt served fresh).

## Release recipe (FR-OPS-17 par-1-fix) — PENDING-FREEZE

`qwen3.8-flash-next` on one Thor, its full native window, eight seats, vision on. The block below is the recipe the
release gates run; the commit is filled in when the release build is frozen and approved.

```
ENGINE_COMMIT=PENDING-FREEZE        # thor/par-1-fix
CHECKPOINT=local-inference-lab/Qwen3.8-Flash-Next-NVFP4@7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd

TENSORFOLD_PREFILL_ROWS=4096
TENSORFOLD_MEMORY_RESERVE_GIB=2.5
TENSORFOLD_THOR_DENSE_WEIGHTS=dequant
TENSORFOLD_VISION_WORKSPACE_MIB=1024
TENSORFOLD_ADMISSION_WAIT_S=30

python -m tensorfold.cli serve "$CHECKPOINT_DIR" --backend cuda --kv-dtype int8 \
  --parallel 8 --context 262144 --vision --vision-max-images 50 --vision-image-tokens 16384
```

Every other setting is its default; the Thor switches below are on by default on sm_110.

## Memory, as built

Thor's GPU allocates from the host's 128 GB. The serving memory is planned once at load, in this order:

1. **The n-gram tables are locked** (`mlock`, `thor_kv.lock_tables`): 26.82 GiB of NVFP4 n-gram rows for this
   checkpoint. Locked pages leave `MemAvailable` (measured 43.72 -> 16.60 GiB) and no allocation can take them, so
   every lookup stays a memory read however the caches grow. The container's `ulimit -l` must allow it.
2. **The n-gram rows are decoded through a pair table** (`host_table.NVFP4Table.use_pairs`): a 256 x 256 table of
   the two values each (scale byte, code byte) pair decodes to, built by the same dequantization the per-call path
   runs, so a row's bits are identical by construction (all 65,536 pairs at five table scales in
   `tests/test_ngram_nvfp4_pairs.py`; 200,000 real rows, 32,000,000 values, identical on Thor). One lookup per byte
   replaces the per-call dequantization's row-sized temporaries, whose 4 KiB-page faults had slowed 8k prompts.
3. **Dense weights stay at their checkpoint size** (`TENSORFOLD_THOR_DENSE_WEIGHTS=dequant`): a prompt GEMM on an
   MXFP8 weight (attention, DeltaNet, shared expert) dequantizes that one weight into an 82 MiB staging buffer before
   its bf16 cuBLAS call, with the same bits as a bf16 copy. Bf16 copies (`copy`) would take 5.5 GiB, which the locked
   tables and eight full-window seats leave no room for. Decode's lane kernels read the MXFP8 weights directly in
   either mode.
4. **One cache pool** (`thor_kv.reserve`, after the warm-ups): `MemAvailable` less the host reserve
   (`TENSORFOLD_MEMORY_RESERVE_GIB=2.5`) and the on-demand decode graphs' budget (1 GiB) is allocated once
   (7.92 GiB measured) and freed into torch's caching allocator. Every cache grow then splits that block instead of
   asking the driver for pages, so serving never waits on the kernel's reclaim or compaction.
5. **The room**: the pool less eight kept snapshots (about 110 MB each) and the image tower's workspace
   (`TENSORFOLD_VISION_WORKSPACE_MIB=1024`): **6.06 GiB** of KV cache. At this checkpoint's int8 KV, about 18.3 KB
   per token, a full 262,144-token window takes 4.47 GiB.

The startup log states the plan, e.g. `every stream's caches from one 7.92 GiB pool reserved at load: 6.06 GiB for
caches past 8 kept snapshots and 1.00 GiB of image tower workspace (262151 tokens, 1.36 full 262151-token
windows); 1.00 GiB left for on-demand graphs and 2.50 GiB for the host`.

## Admission and capacity

Every admitted request is promised its worst case (its caches at the window its prompt plus `max_tokens` can reach)
before it takes a seat, so no running stream is ever ended to make room for another. A request that can never fit
the room is refused at once (HTTP 400, `context_length_exceeded`); one that does not fit now waits in the queue,
and with this recipe's `TENSORFOLD_ADMISSION_WAIT_S=30` a request still waiting after 30 seconds is refused with
HTTP 503, `Retry-After` and a `reason` (`"slots"`: every seat stayed busy; `"memory"`: the room stayed promised).
A streamed request's headers wait for its seat, so the refusal is a plain 503, never an error inside an open stream.
`GET /v1/capacity` reports seats, room, queued and refused requests and the kept conversations; `POST /v1/capacity`
previews one request (its cached prefix, its worst case, whether it is admissible now). The field contract is in
[`docs/api.md`, "Capacity and admission"](../api.md#capacity-and-admission).

Kept conversations live in whatever room the live requests leave. When a grow needs room, a request needs a seat
or the kept points pass their cap, the least recently used idle conversation goes first (a preview counts as a use),
and every conversation freed is counted by cause in `/v1/capacity`, so a client knows before it sends whether its
next turn will run cold.

## Measured capacity

PENDING-FREEZE: measured on the pre-freeze build (dev runs, not gated); the gated release runs restate them.

- **Room:** 6.06 GiB of KV cache = **one request of the full 262,144-token window, or about 355,000 tokens across
  all live and kept conversations** (1.36 full windows). More seats share the same room.
- **Eight dynamic sessions** (`tools/thor/session_stress.py`, 8 sessions x 6 turns, prompts growing to 224,732
  tokens, stress-dev1): every turn completed with no error, killed stream or timeout; 34 of 40 follow-ups resumed
  their cached prefix and the other 6 were announced cold by the preview before they ran; up to **4 kept
  conversations of 32k+ tokens, 1 of 64k+**, were held at once.
- **One long conversation:** an 80k-token prompt-reuse chain resumes 100% of its prefix on every follow-up when it
  has the room to itself (kvpool-dev2).

## Thor switches

| Switch | Default on sm_110 | Off / other value |
| --- | --- | --- |
| `TENSORFOLD_THOR_KV_POOL` | the tables locked and one cache pool | `0`: caches grow from the driver, the wide-piece pool guard, the tables' pages evictable |
| `TENSORFOLD_THOR_NGRAM_PAIRS` | NVFP4 n-gram rows through the pair table | `0`: per-call dequantization |
| `TENSORFOLD_THOR_DENSE_WEIGHTS` | `copy` (bf16 copies of MXFP8 weights) | `dequant`: no copies (this recipe) |
| `TENSORFOLD_THOR_DECODE_GRAPH` | decode rounds replay CUDA graphs | `0`: every round eager |
| `TENSORFOLD_ADMISSION_WAIT_S` | unset: requests wait until admitted | seconds: the bounded wait (this recipe: 30) |

## Fixes in this build

- **DEC-2 hang.** `attn_ptrs()`'s per-stream cache key sampled one layer's live pointer; under 8 concurrent streams
  the allocator could reuse that address for a different stream between calls, so the key matched by coincidence and
  a decode round spun on stale data. The key is now `State.version` (`thor_dec2.py`), which changes only on a real
  cache rebuild.
- **Admission control.** `NoRoom` is a `CapacityError` (HTTP 503, `Retry-After`, `reason`), `admit()` promises a
  stream's worst case before it touches a slot, the bounded wait refuses rather than queues forever, and
  `/v1/capacity` reports the state a client or router needs to back off first.
- **Memory plan.** The locked tables and the one pool above replace the earlier path, where a cache grow sized
  against `MemAvailable` took the mapped tables' pages (a 1.4 GiB grow took 70 s and left 9% of the tables on disk).
