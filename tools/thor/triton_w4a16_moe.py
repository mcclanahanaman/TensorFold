"""A W4A16 grouped NVFP4 expert block in Triton, as a Phase A probe (not wired into TensorFold's engine).

The checkpoint's e2m1 code times its e4m3 scale is exact in bf16 (at most 6 significant bits), so each weight tile is
widened to bf16 in registers and multiplied on the bf16 tensor cores (``tl.dot``, fp32 accumulation); weight_scale_2
multiplies in the epilogue, SwiGLU rounds like TensorFold's (bf16 g and u, bf16 silu(g), bf16 product). Tiles are
swap-AB: 128 weight rows are the MMA's M side, a block of one expert's (token, slot) pairs its N side.

    python tools/thor/triton_w4a16_moe.py --model /hf/... --out /out/triton_w4a16.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import moe_common as mc  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")


@triton.jit
def _e2m1(code):
    """int32 codes (sign in bit 3) -> bf16 values, from the bits (exact)."""

    m = code & 7
    pat = tl.where(m == 0, 0, ((126 + (m >> 1)) << 7) | (tl.where(m >= 2, m & 1, 0) << 6)) | ((code & 8) << 12)
    return pat.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _e4m3(byte):
    """e4m3 bytes (int32) -> fp32 values, exact (NaN codes ignored: scales are finite)."""

    e = (byte >> 3) & 0xF
    m = byte & 0x7
    normal = (((e + 120) << 7) | (m << 4)).to(tl.int32) << 16
    v = tl.where(e == 0, m.to(tl.float32) * (2.0 ** -9), normal.to(tl.float32, bitcast=True))
    return tl.where((byte & 0x80) != 0, -v, v)


@triton.jit
def _tile(CODES, SC, rows, k0, K: tl.constexpr, BW: tl.constexpr, BK: tl.constexpr):
    """Weight rows ``rows`` [BW], inputs k0 .. k0 + BK -> bf16 [BW, BK] = code * e4m3 scale (exact)."""

    c = tl.load(CODES + rows[:, None] * (K // 2) + (k0 // 2 + tl.arange(0, BK // 2))[None, :]).to(tl.int32)
    codes = tl.reshape(tl.join(c & 0xF, c >> 4), (BW, BK))
    s = tl.load(SC + rows[:, None] * (K // 16) + (k0 // 16 + tl.arange(0, BK // 16))[None, :]).to(tl.int32)
    sv = _e4m3(s).to(tl.bfloat16)
    sv = tl.reshape(tl.broadcast_to(sv[:, :, None], (BW, BK // 16, 16)), (BW, BK))
    return _e2m1(codes) * sv


@triton.jit
def _gate_up(X, CODES, SC, S2, PAIRS, BLK_E, BLK_START, BLK_CNT, OUT,
             K: tl.constexpr, NI: tl.constexpr, SLOTS: tl.constexpr, BW: tl.constexpr, BT: tl.constexpr,
             BK: tl.constexpr):
    b = tl.program_id(0)
    cnt = tl.load(BLK_CNT + b)
    if cnt <= 0:
        return
    e = tl.load(BLK_E + b).to(tl.int64)
    start = tl.load(BLK_START + b)
    rt = tl.arange(0, BT)
    ok = rt < cnt
    pair = tl.load(PAIRS + start + rt, mask=ok, other=0)
    tok = (pair // SLOTS).to(tl.int64)
    rw = tl.program_id(1) * BW + tl.arange(0, BW)
    codes = CODES + e * (2 * NI) * (K // 2)
    sc = SC + e * (2 * NI) * (K // 16)
    rk = tl.arange(0, BK)
    accg = tl.zeros((BW, BT), tl.float32)
    accu = tl.zeros((BW, BT), tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(X + tok[None, :] * K + (k0 + rk)[:, None], mask=ok[None, :], other=0.0)   # [BK, BT]
        accg = tl.dot(_tile(codes, sc, rw, k0, K, BW, BK), x, accg)
        accu = tl.dot(_tile(codes, sc, NI + rw, k0, K, BW, BK), x, accu)
    g = (accg * tl.load(S2 + 2 * e)).to(tl.bfloat16).to(tl.float32)
    u = (accu * tl.load(S2 + 2 * e + 1)).to(tl.bfloat16).to(tl.float32)
    h = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32) * u
    tl.store(OUT + pair[None, :].to(tl.int64) * NI + rw[:, None], h.to(tl.bfloat16), mask=ok[None, :])


@triton.jit
def _down(A, CODES, SC, S2, PAIRS, BLK_E, BLK_START, BLK_CNT, OUT,
          K: tl.constexpr, N: tl.constexpr, BW: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr):
    b = tl.program_id(0)
    cnt = tl.load(BLK_CNT + b)
    if cnt <= 0:
        return
    e = tl.load(BLK_E + b).to(tl.int64)
    start = tl.load(BLK_START + b)
    rt = tl.arange(0, BT)
    ok = rt < cnt
    pair = tl.load(PAIRS + start + rt, mask=ok, other=0).to(tl.int64)
    rw = tl.program_id(1) * BW + tl.arange(0, BW)
    codes = CODES + e * N * (K // 2)
    sc = SC + e * N * (K // 16)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BW, BT), tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(A + pair[None, :] * K + (k0 + rk)[:, None], mask=ok[None, :], other=0.0)
        acc = tl.dot(_tile(codes, sc, rw, k0, K, BW, BK), a, acc)
    y = acc * tl.load(S2 + e)
    tl.store(OUT + pair[None, :] * N + rw[:, None], y.to(tl.bfloat16), mask=ok[None, :])


class TritonW4A16:
    def __init__(self, layer: mc.Layer) -> None:
        self.codes1 = torch.cat([layer.gate.w, layer.up.w], 1).contiguous()                      # [E, 2NI, D/2]
        self.sc1 = torch.cat([layer.gate.s, layer.up.s], 1).contiguous().view(torch.uint8)       # [E, 2NI, D/16]
        self.s21 = torch.stack([layer.gate.s2, layer.up.s2], 1).float().contiguous()             # [E, 2]
        self.codes2 = layer.down.w.contiguous()
        self.sc2 = layer.down.s.contiguous().view(torch.uint8)
        self.s22 = layer.down.s2.float().contiguous()
        self.cfg = dict(BT=64, BK=64, BW=128, w1=4, s1=3, w2=4, s2=3)

    def plan(self, ids: torch.Tensor, bt: int):
        """Pairs sorted by expert and blocks of up to ``bt`` pairs of one expert, all on the device (no sync)."""

        flat = ids.flatten()
        pairs = torch.argsort(flat, stable=True).to(torch.int32)
        counts = torch.bincount(flat, minlength=mc.E)
        nblk = (counts + bt - 1) // bt
        cum = torch.cumsum(nblk, 0)
        first_pair = torch.cumsum(counts, 0) - counts
        bmax = -(-flat.numel() // bt) + mc.E
        b = torch.arange(bmax, device=ids.device)
        be = torch.searchsorted(cum, b, right=True).clamp(max=mc.E - 1)
        local = b - (cum - nblk)[be]
        cnt = (counts[be] - local * bt).clamp(0, bt)
        cnt = torch.where(b < cum[-1], cnt, torch.zeros_like(cnt))
        start = (first_pair[be] + local * bt).to(torch.int32)
        return pairs, be.to(torch.int32), start, cnt.to(torch.int32), bmax

    @torch.no_grad()
    def __call__(self, x, ids, w, act=None, y=None):
        m = x.shape[0]
        c = self.cfg
        pairs, be, start, cnt, bmax = self.plan(ids, c["BT"])
        act = torch.empty(m * mc.TOP, mc.NI, dtype=torch.bfloat16, device=x.device) if act is None else act
        y = torch.empty(m * mc.TOP, mc.D, dtype=torch.bfloat16, device=x.device) if y is None else y
        _gate_up[(bmax, mc.NI // c["BW"])](x, self.codes1, self.sc1, self.s21, pairs, be, start, cnt, act,
                                           K=mc.D, NI=mc.NI, SLOTS=mc.TOP, BW=c["BW"], BT=c["BT"], BK=c["BK"],
                                           num_warps=c["w1"], num_stages=c["s1"])
        _down[(bmax, mc.D // c["BW"])](act, self.codes2, self.sc2, self.s22, pairs, be, start, cnt, y,
                                       K=mc.NI, N=mc.D, BW=c["BW"], BT=c["BT"], BK=c["BK"],
                                       num_warps=c["w2"], num_stages=c["s2"])
        return (y.view(m, mc.TOP, mc.D).float() * w[..., None]).sum(1).to(torch.bfloat16)


CONFIGS = [dict(BT=bt, BK=bk, BW=128, w1=wp, s1=st, w2=wp, s2=st)
           for bt in (32, 64, 128) for bk in (64, 128) for wp in (4, 8) for st in (2, 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="/out/triton_w4a16.json")
    ap.add_argument("--layers", default="0,23,47")
    ap.add_argument("--acc-ms", default="128,512,2048,4096")
    ap.add_argument("--speed-ms", default="512,2048,4096,8192")
    ap.add_argument("--speed-layer", type=int, default=23)
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()
    rep = {"triton": triton.__version__, "accuracy": {}, "speed": {}, "errors": {}, "asm": {}}
    out = Path(args.out)

    def save():
        out.write_text(json.dumps(rep, indent=1, default=str))

    ck = mc.Checkpoint(args.model)
    layers = [int(v) for v in args.layers.split(",")]
    order = layers + ([args.speed_layer] if args.speed_layer not in layers else [])
    for li in order:
        layer = ck.layer(li)
        path = TritonW4A16(layer)
        if li == args.speed_layer:
            # a small sweep per M first: the best config then also serves the accuracy check at that M
            for m in [int(v) for v in args.speed_ms.split(",")]:
                x = mc.activations(m, seed=2000 + m)
                ids, w = mc.route(x, layer.router)
                best = None
                for cfg in CONFIGS:
                    path.cfg = cfg
                    try:
                        t = mc.time_cuda(lambda: path(x, ids, w), reps=5, warmup=2)
                    except Exception as exc:  # noqa: BLE001
                        rep["errors"][f"S.M{m}.{cfg}"] = repr(exc)[:500]
                        continue
                    if best is None or t["median_ms"] < best[0]["median_ms"]:
                        best = (t, dict(cfg))
                if best is None:
                    continue
                path.cfg = best[1]
                t = mc.time_cuda(lambda: path(x, ids, w), reps=args.reps)
                t["tflops"] = mc.flops(m) / (t["median_ms"] * 1e-3) / 1e12
                t["cfg"] = best[1]
                rep["speed"][f"M{m}"] = t
                print(f"[{mc.stamp()}] speed M={m} triton-w4a16 {t['median_ms']:.3f} ms {t['tflops']:.2f} TFLOPS "
                      f"cfg {best[1]}", flush=True)
                save()
            try:
                k = _gate_up.cache[torch.cuda.current_device()]
                asm = next(iter(k.values())).asm["ptx"]
                rep["asm"] = {kk: asm.count(kk) for kk in ("tcgen05.mma", "wgmma", "mma.sync", "tcgen05.ld")}
                print("gate_up PTX:", rep["asm"], flush=True)
            except Exception as exc:  # noqa: BLE001
                rep["asm"] = repr(exc)
        if li in layers:
            ref = mc.reference_weights(layer)
            path.cfg = dict(BT=64, BK=64, BW=128, w1=4, s1=3, w2=4, s2=3)
            entry = {}
            for m in [int(v) for v in args.acc_ms.split(",")]:
                x = mc.activations(m, seed=1000 + m)
                ids, w = mc.route(x, layer.router)
                refo = mc.reference(x, ids, w, ref)
                try:
                    got = path(x, ids, w)
                    torch.cuda.synchronize()
                    met = mc.metrics(got.float(), refo)
                except Exception as exc:  # noqa: BLE001
                    rep["errors"][f"L{li}.M{m}"] = f"{exc!r}\n{traceback.format_exc()[-3000:]}"
                    print("accuracy failed", repr(exc), flush=True)
                    save()
                    continue
                entry[f"M{m}"] = met
                print(f"[{mc.stamp()}] L{li} M={m} triton-w4a16 rel {met['rel_fro']:.4e} "
                      f"max|d| {met['max_abs']:.4e} cos {met['cosine']:.8f}", flush=True)
            rep["accuracy"][f"layer{li}"] = entry
            save()
            del ref
        del path, layer
        torch.cuda.empty_cache()
    save()
    print("done; errors:", len(rep["errors"]), flush=True)


if __name__ == "__main__":
    main()
