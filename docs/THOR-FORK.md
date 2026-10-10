# TensorFold Thor fork

This repository is TensorFold's Python engine line with Jetson AGX Thor (`sm_110`) support. It is a fork, not an
upstream TensorFold release.

On Thor the Flash Next CUDA family adds:

- NVFP4 expert kernels (FlashInfer's W4A16 grouped GEMM, with decode graphs recorded at load)
- dense, DeltaNet and sparse-attention prompt kernels
- decode CUDA graphs and the device-fed MTP draft chain
- n-gram tables read from NVMe through a row cache
- one KV pool with admission control (HTTP 503 and `Retry-After`) and `GET` / `POST /v1/capacity`
- vision host-memory budgeting

The serving recipe for this build is [docs/recipes/thor-sm110.md](recipes/thor-sm110.md).

## Status

This fork branched from TensorFold 0.6.3. Upstream's Python engine is frozen at 0.6.6 and is being retired. Native-engine Flash Next CUDA work is tracked
upstream in [ashhart/TensorFold#472](https://github.com/ashhart/TensorFold/issues/472). This fork is not an upstream
release.

## Switches

Every `TENSORFOLD_THOR_*` variable found in this tree is listed once. On `sm_110` a feature is on unless its line says
otherwise. `0` (also `off`, `false`, `no`) is that feature's kill switch: the earlier path, the same accepted output.

### Experts, dense prompt GEMMs, DeltaNet, sparse attention

- `TENSORFOLD_THOR_FI_MOE` — `0` keeps the portable NVFP4 expert kernel (FlashInfer's W4A16 grouped GEMM is the default).
- `TENSORFOLD_THOR_FI_MOE_GRAPHS` — `0` launches those expert GEMMs on every call instead of replaying a CUDA graph.
- `TENSORFOLD_THOR_FI_MOE_PRERECORD` — `0` records a decode buffer set's expert graphs on first use instead of at load.
- `TENSORFOLD_THOR_DENSE` — `0` keeps the portable prompt GEMMs (cuBLAS bf16 is the default).
- `TENSORFOLD_THOR_DENSE_WEIGHTS` — `copy` (default) keeps bf16 copies of MXFP8 weights; `dequant` dequantizes at each call.
- `TENSORFOLD_THOR_QMMF_SLICES` — `0` keeps the cluster reduction for short MXFP8 K-slice GEMMs.
- `TENSORFOLD_THOR_QMMF_ROWS48` — `0` keeps 64-row tiles for MXFP8 windows of 33–48 rows.
- `TENSORFOLD_THOR_GDN` — `0` keeps TensorFold's DeltaNet prompt chain (FlashInfer's chunked kernel is the default).
- `TENSORFOLD_THOR_GDN_CHAIN` — `0` keeps the tree kernel for DeltaNet chain windows; `1` or `2` force a chain-kernel variant.
- `TENSORFOLD_THOR_QSA` — `0` keeps TensorFold's sparse-attention prompt kernels.
- `TENSORFOLD_THOR_QSA_ATTN` — `0` keeps the earlier sparse-attention kernels (chunk partials, merge, TensorFold's selection).
- `TENSORFOLD_THOR_QSA_SEL` — `0` keeps the Triton block selection (the CUDA radix select is the default).
- `TENSORFOLD_THOR_SHX_FUSE` — `0` keeps torch's shared-expert elementwise kernels.
- `TENSORFOLD_THOR_HC_FUSE` — `0` keeps TensorFold's hyper-connection read-out (the fused prompt kernels are the default).
- `TENSORFOLD_THOR_HC_FUSE2` — `0` keeps the Triton hyper-connection kernels (warp-specialized kernels are the default from 256 rows).
- `TENSORFOLD_THOR_MOE_GLUE` — `0` keeps the expert scatter and the shared expert's separate kernels.

### Decode graphs and the MTP draft chain

- `TENSORFOLD_THOR_DECODE_GRAPH` — `0` runs every decode round eagerly.
- `TENSORFOLD_THOR_DECODE_GRAPH_CAP` — the most decode graphs kept (past it, new keys stay eager).
- `TENSORFOLD_THOR_DECODE_GRAPH_STREAMS` — graph only forwards of at most this many streams.
- `TENSORFOLD_THOR_DECODE_GRAPH_ROWS` — which verification windows are recorded at startup (`2-W` by default; `2,W` records only the ends).
- `TENSORFOLD_THOR_DECODE_GRAPH_STARTUP` — `0` records every startup key on its second forward instead of at load.
- `TENSORFOLD_THOR_DECODE_GRAPH_LAZY` — `1` records keys outside the startup set on their second forward.
- `TENSORFOLD_THOR_DECODE_GRAPH_DEMAND` — `0` leaves those later one-stream keys eager.
- `TENSORFOLD_THOR_DECODE_GRAPH_DEMAND_MIB` — MiB budget for on-demand graphs (default 1024).
- `TENSORFOLD_THOR_DECODE_GRAPH_HOST` — `0` charges an on-demand graph's device memory only, not the driver's host bytes.
- `TENSORFOLD_THOR_DECODE_GRAPH_CAPTURE` — `0` runs the startup warm-ups eagerly and records no graphs (diagnostics).
- `TENSORFOLD_THOR_DECODE_GRAPH_MULTI` — `0` keeps concurrent forwards eager; `all` also graphs verification forwards and first MTP steps.
- `TENSORFOLD_THOR_DECODE_SPIN` — `0` lets the host sleep while it waits for the GPU in decode.
- `TENSORFOLD_THOR_MTP_FED` — `0` keeps the host-fed MTP draft chain (the device-fed chain is the default when decode graphs are on).
- `TENSORFOLD_THOR_MTP_FED_LOG` — file for the device-fed chain's counts at exit (stderr when unset).

### KV pool, n-gram tables, prompt pieces

- `TENSORFOLD_THOR_KV_POOL` — `0` grows caches from the driver, with the wide-piece pool guard and evictable table pages.
- `TENSORFOLD_THOR_KV_POOL_FIT` — `0` sizes the pool once from `MemAvailable` before the reservation (no refit).
- `TENSORFOLD_THOR_KV_POOL_TOUCH` — `0` leaves the pool's pages untouched until first use (diagnostics).
- `TENSORFOLD_THOR_POOL` — `0` keeps the earlier memory gate for wide prompt pieces (no kept ends freed to protect the pool).
- `TENSORFOLD_THOR_MEM_PROBE` — path of a trigger file; memory breakdowns print at startup and when the file appears (diagnostics).
- `TENSORFOLD_THOR_NGRAM_NVME` — `0` keeps the n-gram tables in memory (the kill switch). `1`, or `--ple-on-ssd` on `sm_110`, reads them from NVMe.
- `TENSORFOLD_THOR_NGRAM_CACHE_GIB` — row-cache size in GiB (default 0.53; 0.35–1.4 is allowed).
- `TENSORFOLD_THOR_NGRAM_FILE` — repacked row file used instead of the checkpoint shards (one 90-byte record a row).
- `TENSORFOLD_THOR_NGRAM_IO` — `aio` (default) or `threads`.
- `TENSORFOLD_THOR_NGRAM_DEPTH` — reads in flight on a reader (default 64).
- `TENSORFOLD_THOR_NGRAM_HELPERS` — extra reader threads for prompt-sized bursts (default 3).
- `TENSORFOLD_THOR_NGRAM_CPU` — threads for the row cache's CPU-side work (default 8).
- `TENSORFOLD_THOR_NGRAM_PAGE` — bytes in one read (default 4096).
- `TENSORFOLD_THOR_NGRAM_PROBATION` — probation ways out of 16 (default 4; `0` is CLOCK over every way).
- `TENSORFOLD_THOR_NGRAM_INIT` — use count a new row starts with (default 0).
- `TENSORFOLD_THOR_NGRAM_ARRIVAL` — `0` turns off the low-priority reads of a prompt's first rows ahead of its stage.
- `TENSORFOLD_THOR_NGRAM_ARRIVAL_THREADS` — readers for those arrival reads (default 2).
- `TENSORFOLD_THOR_NGRAM_ARRIVAL_WAVE` — arrival reads in flight on each of those readers (default 64).
- `TENSORFOLD_THOR_NGRAM_ARRIVAL_ROWS` — prompt rows read ahead (default 8192).
- `TENSORFOLD_THOR_NGRAM_PAIRS` — `0` dequantizes an NVFP4 row on each call (the pair table is the default).
- `TENSORFOLD_THOR_NGRAM_SNAPSHOT` — path of the row-cache snapshot restored at start and saved at exit (unset: none).
- `TENSORFOLD_THOR_NGRAM_SNAPSHOT_S` — seconds between snapshot saves (default 1800; `0` saves only at exit).
- `TENSORFOLD_THOR_NVME_EXT_DIR` — build directory for the NVMe row-cache extension (`TORCH_EXTENSIONS_DIR` when unset).
- `TENSORFOLD_THOR_DEC2_HOST` — `0` keeps the earlier decode host path.
- `TENSORFOLD_THOR_DEC2_POOL_ROWS` — gathers of at least this many rows are split across the table's pool (default 384).
- `TENSORFOLD_THOR_DEC2_SPLIT` — minimum rows in one of those pieces (default 32; `0` only maps the pages).
- `TENSORFOLD_THOR_DEC2_AHEAD` — `0` turns off gathering the next window's known rows while an MTP step runs.
- `TENSORFOLD_THOR_DEC2_TABLES` — `0` builds the decode step's device tables on the earlier path.
- `TENSORFOLD_THOR_DEC2_CHECK` — `1` checks reused window ids and fast pointer tables against the earlier computation (diagnostics, off by default).
- `TENSORFOLD_THOR_STAGE_AHEAD` — `0` gathers a prompt's n-gram rows after `stage`'s wait (the default gathers them before the wait).

### Vision

- `TENSORFOLD_THOR_VISION` — `0` leaves the vision tower on transformers' forward.
- `TENSORFOLD_THOR_VISION_ROPE` — `0` keeps transformers' rotary passes.
- `TENSORFOLD_THOR_VISION_VARLEN` — `0` keeps transformers' per-segment attention calls.
- `TENSORFOLD_THOR_VISION_PREP` — `0` scales and patches video frames on the CPU path.
- `TENSORFOLD_THOR_VISION_STREAM` — `0` runs a video's tower calls after the frames have finished decoding.
- `TENSORFOLD_THOR_VISION_MEM` — `0` keeps the whole vision workspace in the pool (16 preparation slots, patch arrays kept until the reply ends).
- `TENSORFOLD_THOR_VISION_HOST_MIB` — host part of the vision workspace, in MiB (unset: the workspace minus the tower's GPU part). The pool still leaves max(that part, 960 MiB) out.

## How it was validated

Replies were bit-identical to the previous build at every step. Decode, prefill, vision, soak and stress runs were
gated, and a text-quality suite was run. The numbers live in the recipe repository:
[github.com/mcclanahanaman/Qwen3.8-Flash-Next-Jetson-AGX-Thor-TensorFold](https://github.com/mcclanahanaman/Qwen3.8-Flash-Next-Jetson-AGX-Thor-TensorFold).

## Credits

TensorFold is by Ash Hart, Apache-2.0, upstream
[ashhart/TensorFold](https://github.com/ashhart/TensorFold). Mia AI Lab's recipes (MIT) are the serving and benchmark
protocol. Thor kernels and engine work are by Brendan McClanahan with Claude (Anthropic) and Grok (xAI). Helper
modules are by GLM-5.3-Flash workers.

## License

Apache-2.0. Modified files carry the upstream license.
