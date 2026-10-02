// Flash Next's QSA block selection for prompt rows on Jetson Thor (sm_110) (THOR-K4-QSA-SEL): attention._select's
// lists, bit for bit.
//
// A sparse row lists its TOP highest-scoring complete 4-key blocks in block order (score ties to lower block ids),
// then its unfinished tail. attention._select finds the cut, the largest key t with at least TOP keys >= t, by 32
// halvings over the row's order-preserving 32-bit keys; that t is the TOP-th largest key, which a radix select finds
// in three steps. One CTA a row: the row's keys are held in registers (KPT a thread); each step counts the digits (B0
// bits, then the rest of the key in two digits) of the keys that share the cut's bits found so far in a shared-memory
// histogram and keeps the digit where the count from the top reaches the cut's rank; then every block above the cut
// and the first `need` blocks equal to it are placed in block order by warp ballots and a scan over the CTA's warps. A
// row wider than THREADS x KPT blocks is read again from global memory (L2), a tile of that width at a time, for each
// step.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr unsigned FULL = 0xffffffffu;

// attention._select's key: float order as unsigned order (negatives flipped, the sign bit set on the rest)
__device__ __forceinline__ uint32_t key_of(float v) {
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// step P's digit: bits [SHIFT, SHIFT + BITS) of the key; B0 bits first, then the other 32 - B0 in two digits
template <int B0, int P> struct Digit {
    static constexpr int B1 = (33 - B0) / 2, B2 = 32 - B0 - B1;
    static constexpr int BITS = P == 0 ? B0 : P == 1 ? B1 : B2;
    static constexpr int SHIFT = P == 0 ? 32 - B0 : P == 1 ? B2 : 0;
};

// a warp's 32 x KPT blocks of a tile: lane l's key i is block base + 32 i + l (coalesced loads)
template <int KPT>
__device__ __forceinline__ void load_tile(uint32_t (&key)[KPT], const float* __restrict__ row, int base, int complete,
                                          int lane) {
#pragma unroll
    for (int i = 0; i < KPT; ++i) {
        const int b = base + 32 * i + lane;
        key[i] = b < complete ? key_of(__ldg(row + b)) : 0u;
    }
}

// step P's histogram: the digits of the keys whose higher bits equal the cut's so far (every key at step 0)
template <int B0, int P, int KPT>
__device__ __forceinline__ void count(int* hist, const uint32_t (&key)[KPT], uint32_t prefix, int base, int complete,
                                      int lane) {
    using D = Digit<B0, P>;
    constexpr int SH = D::SHIFT, MASK = (1 << D::BITS) - 1, HI = SH + D::BITS;
#pragma unroll
    for (int i = 0; i < KPT; ++i) {
        bool in = base + 32 * i + lane < complete;
        if constexpr (P > 0) in = in && (key[i] >> HI) == (prefix >> HI);
        if (in) atomicAdd(&hist[(key[i] >> SH) & MASK], 1);
    }
}

// the cut's digit at this step: the largest d with at least `need` counted keys of digit >= d; found = {d, the cut's
// rank among the keys of digit d}. Thread t sums bins t * BPT .. t * BPT + BPT - 1; a suffix scan over the threads.
template <int THREADS, int BINS>
__device__ __forceinline__ void find(const int* hist, int* wsum, int* found, int need, int w, int lane) {
    constexpr int NW = THREADS / 32, BPT = BINS >= THREADS ? BINS / THREADS : 1;
    const int t = threadIdx.x;
    const bool has = t * BPT < BINS;
    int s = 0;
    if (has) {
#pragma unroll
        for (int j = 0; j < BPT; ++j) s += hist[t * BPT + j];
    }
    int x = s;                                       // inclusive suffix sum within the warp
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_down_sync(FULL, x, o);
        if (lane + o < 32) x += y;
    }
    if (lane == 0) wsum[w] = x;
    __syncthreads();
    int after = 0;
    for (int v = w + 1; v < NW; ++v) after += wsum[v];
    const int incl = x + after, excl = incl - s;
    if (has && excl < need && need <= incl) {        // exactly one thread: the counts from the top cross need here
        int run = excl;
        for (int j = BPT - 1; j >= 0; --j) {
            const int c = hist[t * BPT + j];
            if (run + c >= need) {
                found[0] = t * BPT + j;
                found[1] = need - run;
                break;
            }
            run += c;
        }
    }
    __syncthreads();
}

template <int B0, int P, int THREADS, int KPT>
__device__ __forceinline__ void step(int* hist, int* wsum, int* found, uint32_t (&key)[KPT], const float* row,
                                     int complete, int tiles, uint32_t& prefix, int& need, int w, int lane) {
    constexpr int TILE = THREADS * KPT, BINS = 1 << Digit<B0, P>::BITS;
    for (int i = threadIdx.x; i < BINS; i += THREADS) hist[i] = 0;
    __syncthreads();
    for (int t = 0; t < tiles; ++t) {
        const int base = t * TILE + w * 32 * KPT;
        if (tiles > 1) load_tile<KPT>(key, row, base, complete, lane);
        count<B0, P, KPT>(hist, key, prefix, base, complete, lane);
    }
    __syncthreads();
    find<THREADS, BINS>(hist, wsum, found, need, w, lane);
    prefix |= uint32_t(found[0]) << Digit<B0, P>::SHIFT;
    need = found[1];
}

// STOP (profiling only, wrong lists): 1 returns after step 0, 2 after step 2; 0 is the selection
template <int THREADS, int KPT, int B0, int STOP>
__global__ void __launch_bounds__(THREADS)
select_rows(const float* __restrict__ sc, const int* __restrict__ pos0, int* __restrict__ ids, int* __restrict__ nkr,
            int* __restrict__ spr, int64_t sc_stride, int64_t idw, int ratio, int top) {
    constexpr int NW = THREADS / 32, SEG = 32 * KPT, TILE = THREADS * KPT;
    static_assert(THREADS % 32 == 0 && B0 >= 11 && B0 <= 13, "whole warps; 11 to 13 bits first");
    __shared__ int hist[1 << B0];
    __shared__ int wsum[NW], wa[NW], we[NW];
    __shared__ int found[2];
    const int r = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int end = __ldg(pos0) + r + 1;
    const int complete = end / ratio;
    if (complete <= top) {                           // a dense row: only its length (the whole CTA leaves)
        if (threadIdx.x == 0) {
            nkr[r] = end;
            spr[r] = 0;
        }
        return;
    }
    const float* row = sc + int64_t(r) * sc_stride;
    int* out = ids + int64_t(r) * idw;
    const int tiles = (complete + TILE - 1) / TILE;
    uint32_t key[KPT];
    if (tiles == 1) load_tile<KPT>(key, row, w * SEG, complete, lane);
    uint32_t prefix = 0;                             // the cut's bits found so far
    int need = top;                                  // the cut's rank among the keys that share them
    step<B0, 0, THREADS, KPT>(hist, wsum, found, key, row, complete, tiles, prefix, need, w, lane);
    if constexpr (STOP == 1) {
        if (threadIdx.x == 0) nkr[r] = int(prefix);
        return;
    }
    step<B0, 1, THREADS, KPT>(hist, wsum, found, key, row, complete, tiles, prefix, need, w, lane);
    step<B0, 2, THREADS, KPT>(hist, wsum, found, key, row, complete, tiles, prefix, need, w, lane);
    if constexpr (STOP == 2) {
        if (threadIdx.x == 0) nkr[r] = int(prefix) + need;
        return;
    }
    const uint32_t cut = prefix;                     // need: how many keys equal to the cut are taken (lowest first)
    int eq_carry = 0, placed = 0;                    // equal keys seen and blocks placed in earlier tiles
    for (int t = 0; t < tiles && placed < top; ++t) {
        const int base = t * TILE + w * SEG;
        if (tiles > 1) load_tile<KPT>(key, row, base, complete, lane);
        int na = 0, ne = 0;
#pragma unroll
        for (int i = 0; i < KPT; ++i) {
            const bool ok = base + 32 * i + lane < complete;
            na += __popc(__ballot_sync(FULL, ok && key[i] > cut));
            ne += __popc(__ballot_sync(FULL, ok && key[i] == cut));
        }
        if (lane == 0) {
            wa[w] = na;
            we[w] = ne;
        }
        __syncthreads();
        int eq_tile = 0, chosen_tile = 0, eq_before = 0, at = 0, mine = 0;
        for (int v = 0; v < NW; ++v) {               // warps in block order: equal keys, then chosen blocks before
            const int e = we[v];
            const int c = wa[v] + min(max(need - (eq_carry + eq_tile), 0), e);
            if (v == w) {
                eq_before = eq_carry + eq_tile;
                at = placed + chosen_tile;
                mine = c;
            }
            eq_tile += e;
            chosen_tile += c;
        }
        __syncthreads();                             // wa, we read by every warp before the next tile writes them
        if (mine > 0) {
            int eq_seen = eq_before;
#pragma unroll
            for (int i = 0; i < KPT; ++i) {
                const int b = base + 32 * i + lane;
                const bool ok = b < complete;
                const bool above = ok && key[i] > cut, equal = ok && key[i] == cut;
                const unsigned em = __ballot_sync(FULL, equal);
                const int rank = eq_seen + __popc(em & ((2u << lane) - 1u));       // 1-based, block order
                const bool chosen = above || (equal && rank <= need);
                const unsigned cm = __ballot_sync(FULL, chosen);
                if (chosen) {
                    const int p = at + __popc(cm & ((1u << lane) - 1u));
                    for (int k = 0; k < ratio; ++k) out[p * ratio + k] = b * ratio + k;
                }
                eq_seen += __popc(em);
                at += __popc(cm);
            }
        }
        eq_carry += eq_tile;
        placed += chosen_tile;
    }
    const int tail = end - ratio * complete;         // the unfinished block's keys
    if (threadIdx.x < tail) out[top * ratio + threadIdx.x] = ratio * complete + threadIdx.x;
    if (threadIdx.x == 0) {
        nkr[r] = top * ratio + tail;
        spr[r] = 1;
    }
}

struct Args {
    const float* sc;
    const int* pos0;
    int* ids;
    int* nk;
    int* sp;
    int64_t sc_stride, idw;
    int ratio, top, rows;
    cudaStream_t stream;
};

template <int THREADS, int KPT, int B0, int STOP>
void launch1(const Args& a) {
    select_rows<THREADS, KPT, B0, STOP><<<a.rows, THREADS, 0, a.stream>>>(a.sc, a.pos0, a.ids, a.nk, a.sp, a.sc_stride,
                                                                          a.idw, a.ratio, a.top);
}

template <int THREADS, int KPT>
void launch(const Args& a, int b0, int stop) {
#ifdef THOR_SELECT_TUNING
    if (stop == 1) return b0 == 13 ? launch1<THREADS, KPT, 13, 1>(a) : launch1<THREADS, KPT, 11, 1>(a);
    if (stop == 2) return b0 == 13 ? launch1<THREADS, KPT, 13, 2>(a) : launch1<THREADS, KPT, 11, 2>(a);
    if (b0 == 13) return launch1<THREADS, KPT, 13, 0>(a);
#endif
    launch1<THREADS, KPT, 11, 0>(a);
}

}  // namespace

// rows' lists from scores [>= rows, >= blocks] (fp32, row stride sc.stride(0)) into ids [rows, idw], nk, sparse;
// blocks is the widest row's block count (it picks the CTA size and keys a thread). variant (tuning builds only):
// bit 1 a 13-bit first digit, bits 4-7 a fixed (threads, keys) config, bits 8-9 a profiling stop.
void thor_select_cuda(const at::Tensor& scores, const at::Tensor& pos0, at::Tensor& ids, at::Tensor& nk,
                      at::Tensor& sparse, int64_t rows, int64_t blocks, int64_t ratio, int64_t top, int64_t variant) {
    if (rows <= 0) return;
    const Args a{scores.data_ptr<float>(), pos0.data_ptr<int>(), ids.data_ptr<int>(), nk.data_ptr<int>(),
                 sparse.data_ptr<int>(), scores.stride(0), ids.stride(0), int(ratio), int(top), int(rows),
                 at::cuda::getCurrentCUDAStream()};
    const int b0 = (variant & 2) ? 13 : 11, stop = int((variant >> 8) & 3);
    int cfg = int((variant >> 4) & 15);
    if (cfg == 0) cfg = blocks <= 1024 ? 1 : blocks <= 2048 ? 2 : blocks <= 4096 ? 3 : blocks <= 8192 ? 4
                        : blocks <= 16384 ? 5 : 6;
    switch (cfg) {
        case 1: launch<128, 8>(a, b0, stop); break;
        case 2: launch<128, 16>(a, b0, stop); break;
        case 3: launch<256, 16>(a, b0, stop); break;
        case 4: launch<256, 32>(a, b0, stop); break;
        case 5: launch<512, 32>(a, b0, stop); break;
        case 6: launch<512, 64>(a, b0, stop); break;
#ifdef THOR_SELECT_TUNING
        case 7: launch<1024, 32>(a, b0, stop); break;
        case 8: launch<1024, 16>(a, b0, stop); break;
        case 9: launch<512, 16>(a, b0, stop); break;
        case 10: launch<256, 64>(a, b0, stop); break;
        case 11: launch<1024, 8>(a, b0, stop); break;
#endif
        default: TORCH_CHECK(false, "thor_select: unknown config ", cfg);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
