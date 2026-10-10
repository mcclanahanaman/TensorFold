# Jetson AGX Thor (sm_110)

The Qwen3.8 Flash Next CUDA family runs on Thor behind a capability check (`sm_110`); every path on this page is
inert on other GPUs. Thor-specific kernels route MoE GEMMs through FlashInfer's W4A16 grouped kernel and DeltaNet's
prompt chains through FlashInfer's chunked kernel; sparse-attention prompt scoring, the shared-expert elementwise tail,
hyper-connection fusion and the n-gram table's host-side gather are TensorFold's own Thor paths. Each has a
`TENSORFOLD_THOR_*` switch back to the portable path (listed below); none changes accepted output, only which device
code produces it (the contract every CUDA family keeps: drafted output equals serial output, a resumed prompt equals
that prompt served fresh).

## Release recipe

`qwen3.8-flash-next` on one Thor, its full native window, eight seats, vision on. The block below is the recipe the
release gates run.

```
ENGINE_COMMIT=the branch's release tag thor-release-2026-10-10
CHECKPOINT=local-inference-lab/Qwen3.8-Flash-Next-NVFP4@7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd

TENSORFOLD_PREFILL_ROWS=4096
TENSORFOLD_MEMORY_RESERVE_GIB=2.5
TENSORFOLD_THOR_KV_POOL=1
TENSORFOLD_THOR_DENSE_WEIGHTS=copy
TENSORFOLD_THOR_NGRAM_NVME=1
TENSORFOLD_THOR_NGRAM_FILE=/hf/derived/ngram-7c4f1bc1/ngram.rows
TENSORFOLD_THOR_NGRAM_SNAPSHOT=/cache/ngram-snap/ngram-7c4f1bc1.snap
TENSORFOLD_THOR_NGRAM_SNAPSHOT_S=0
TENSORFOLD_VISION_WORKSPACE_MIB=1024
TENSORFOLD_ADMISSION_WAIT_S=30
NUMPY_MADVISE_HUGEPAGE=0
GLIBC_TUNABLES=glibc.malloc.mmap_threshold=524288
MIMALLOC_PURGE_DELAY=0
MIMALLOC_ABANDONED_PAGE_PURGE=1

python -m tensorfold.cli serve "$CHECKPOINT_DIR" --backend cuda --kv-dtype int8 \
  --parallel 8 --context 262144 --vision --vision-max-images 50 --vision-image-tokens 16384
```

Every other setting is its default; the Thor switches below are on by default on sm_110. The n-gram row file is built once from the checkpoint with tools/thor/ngram_repack.py; /hf and /cache are the container mounts of the Hugging Face cache and a writable cache directory.

## Memory, as built

Thor's GPU allocates from the host's 128 GB. The n-gram tables stay on NVMe and are read through a locked row cache, so
that memory is free for the KV cache. The pool reserves the host's free memory less the host reserve (2.5 GiB,
`TENSORFOLD_MEMORY_RESERVE_GIB`), the on-demand graphs' budget (1 GiB) and the vision allowance kept out of the pool
(0.94 GiB, that is max(the workspace's host part, 960 MiB)). The caches' room subtracts only the 0.62 GiB GPU part of
the 1 GiB vision workspace. At startup the pool is refit after the application's init so the host's idle free memory
lands on its design point: the reserve plus the graph budget plus the vision allowance, about 4.44 GiB. Swap that went
to disk counts as used.

The startup log states the plan:

```
[tensorfold] Thor (sm_110): vision workspace split 0.62 GiB tower (in the pool) + 0.38 GiB host buffers (out of it; the pool leaves 0.94 GiB out for serving vision), a request holding its host bound from its decode to its encode, its arrays dropped and glibc trimmed after (TENSORFOLD_THOR_VISION_MEM=0: the earlier path)
[tensorfold] Thor (sm_110): every stream's caches from one 26.96 GiB pool reserved at load: 25.47 GiB for caches past 8 kept snapshots and 0.62 GiB of image tower workspace (262151 tokens, 5.70 full 262151-token windows); 1.00 GiB left for on-demand graphs, 0.94 GiB for vision requests' host buffers (THOR-VISION-MEM) and 2.50 GiB for the host (TENSORFOLD_MEMORY_RESERVE_GIB); idle free memory 4.49 GiB, +55 MiB from its design point 4.44 GiB after 2 reading(s) (pools 27.75, 26.96 GiB, 1064 MiB in swap on disk counted as used; TENSORFOLD_THOR_KV_POOL_FIT=0 sizes once) (TENSORFOLD_THOR_KV_POOL=0: the earlier path)
[tensorfold] Thor (sm_110): the pool refit after the server's init: every stream's caches from one 26.87 GiB pool: 25.38 GiB for caches (262151 tokens); idle free memory 4.46 GiB, +27 MiB from its design point 4.44 GiB after 2 reading(s) (pools 26.96, 26.87 GiB, 1062 MiB in swap on disk counted as used; TENSORFOLD_THOR_KV_POOL_FIT=0 sizes once) (TENSORFOLD_THOR_KV_POOL=0: the earlier path)
[tensorfold] serving qwen3.8-flash-next at http://127.0.0.1:8010/v1 on CUDA (sampling: temperature 1.0, top_k 20, top_p 0.95; drafts: on; prompts: bf16 activations; context: 262144; loaded in 245.8s)
```

Measured on the release build: KV room 25.3 GiB (about 1.49M tokens, 5 full 262,144-token windows), idle free memory
on the design point. Under the vision MATCH set, with the gate's client included, `MemAvailable` never went under
2.5 GiB (lowest 2.9 GiB).

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

On the release build the KV room is 25.3 GiB (about 1.49M tokens, 5 full 262,144-token windows). Idle free memory
lands on the design point (about 4.44 GiB). Under the vision MATCH set, with the gate's client included,
`MemAvailable` never went under 2.5 GiB (lowest 2.9 GiB). Gated run numbers beyond that live in the recipe repository.

## Thor switches

| Switch | Default on sm_110 | Off / other value |
| --- | --- | --- |
| `TENSORFOLD_THOR_KV_POOL` | one cache pool (NVMe tables stay on disk; their row cache is locked) | `0`: caches grow from the driver, the wide-piece pool guard, in-memory tables' pages evictable |
| `TENSORFOLD_THOR_NGRAM_PAIRS` | NVFP4 n-gram rows through the pair table | `0`: per-call dequantization |
| `TENSORFOLD_THOR_DENSE_WEIGHTS` | `copy` (bf16 copies of MXFP8 weights) | `dequant`: no copies (this recipe) |
| `TENSORFOLD_THOR_DECODE_GRAPH` | decode rounds replay CUDA graphs | `0`: every round eager |
| `TENSORFOLD_THOR_VISION_MEM` | vision host memory budgeted (the workspace split into the tower's GPU part, kept in the pool, and a host part left out of it; a request holds its bound until its encode; its patch arrays dropped and a trim after it) | `0`: the earlier path (the whole workspace in the pool, 16 preparation slots, arrays kept until the reply ends) |
| `TENSORFOLD_THOR_VISION_HOST_MIB` | unset: the host part is the workspace less the tower's GPU part | a MiB number: the host part (`0` up to the workspace), the GPU part is the rest |
| `TENSORFOLD_ADMISSION_WAIT_S` | unset: requests wait until admitted | seconds: the bounded wait (this recipe: 30) |

## Fixes in this build

- **DEC-2 hang.** `attn_ptrs()`'s per-stream cache key sampled one layer's live pointer; under 8 concurrent streams
  the allocator could reuse that address for a different stream between calls, so the key matched by coincidence and
  a decode round spun on stale data. The key is now `State.version` (`thor_dec2.py`), which changes only on a real
  cache rebuild.
- **Admission control.** `NoRoom` is a `CapacityError` (HTTP 503, `Retry-After`, `reason`), `admit()` promises a
  stream's worst case before it touches a slot, the bounded wait refuses rather than queues forever, and
  `/v1/capacity` reports the state a client or router needs to back off first.
- **Memory plan.** The pool above replaces the earlier path, where a cache grow sized
  against `MemAvailable` took the mapped tables' pages (a 1.4 GiB grow took 70 s and left 9% of the tables on disk).
