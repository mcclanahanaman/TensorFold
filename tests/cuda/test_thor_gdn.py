"""Jetson Thor's chunked DeltaNet prompt chain (THOR-K4): FlashInfer's kernel against TensorFold's chain and an fp64
reference, its bits on the 64-row grid, the 16-bit front and back, and Flash Next's prompts, resumes, drafts and packed
passes keeping their bits with the path on, with the served row limit and with one the prompts cross."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.kernels import thor_gdn  # noqa: E402

if not thor_gdn.available():
    pytest.skip("Jetson Thor (sm_110) with FlashInfer's chunked GDN kernel only", allow_module_level=True)

from test_flashnext_forward import V, _model, _state  # noqa: E402
from test_flashnext_gdn_io import _chain_windows, _front, _inputs  # noqa: E402

from tensorfold.cuda.kernels import gdn as tf_gdn  # noqa: E402
from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import gdn, gdn_io  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import (  # noqa: E402
    Engine, entry_end, mtp_decode, prefill, serial_decode)
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402

DEV = "cuda"
HK, HV = 16, 48


@pytest.fixture(params=[None, 128], ids=["limit-served", "limit-128"])
def thor(request, monkeypatch):
    """The path on; ``limit-128``: a prompt's rows from 128 on run TensorFold's chain (passes straddle the limit)."""

    if request.param is not None:
        monkeypatch.setattr(thor_gdn, "LIMIT", request.param)
    assert thor_gdn.prepare(HK, HV) and thor_gdn.use()
    yield
    thor_gdn.set_active(False)


def _rows(rows: int, seed: int):
    """Chain inputs like the front's: q, k fp32 L2-normed (q scaled), v bf16, g in (0, 1), beta = bf16(sigmoid)."""

    g = torch.Generator(device=DEV).manual_seed(seed)

    def normed(x):
        x = torch.nn.functional.silu(x).bfloat16().float()
        return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)

    q = normed(torch.randn((rows, HK, 128), generator=g, device=DEV)) / 128 ** 0.5
    k = normed(torch.randn((rows, HK, 128), generator=g, device=DEV))
    v = torch.nn.functional.silu(1.5 * torch.randn((rows, HV, 128), generator=g, device=DEV)).bfloat16()
    decay = torch.logspace(-3, 0, HV, device=DEV)
    gate = torch.exp(-decay * torch.nn.functional.softplus(torch.randn((rows, HV), generator=g, device=DEV)))
    beta = torch.sigmoid(torch.randn((rows, HV), generator=g, device=DEV)).bfloat16().float()
    state = torch.randn((HV, 128, 128), generator=g, device=DEV) * 0.05
    return q.contiguous(), k.contiguous(), v.contiguous(), gate.contiguous(), beta.contiguous(), state


def _reference(q, k, v, gate, beta, state):
    """The rule in fp64, one row at a time."""

    rep = HV // HK
    qd, kd = q.double().repeat_interleave(rep, 1), k.double().repeat_interleave(rep, 1)
    s, out = state.double().clone(), []
    for t in range(q.shape[0]):
        s.mul_(gate[t].double()[:, None, None])
        delta = (v[t].double() - torch.einsum("hvk,hk->hv", s, kd[t])) * beta[t].double()[:, None]
        s.add_(delta[:, :, None] * kd[t][:, None, :])
        out.append(torch.einsum("hvk,hk->hv", s, qd[t]))
    return torch.stack(out), s


def _rel(a, ref):
    return float((a.double() - ref).norm() / ref.norm())


def _thor(q, k, v, gate, beta, state):
    final = torch.empty_like(state)
    y = thor_gdn.chain(q.half(), k.half(), v.half(), gate, beta, state, final)
    return y, final


def test_the_chain_reads_out_as_exactly_as_tensorfolds(thor):
    """fp16 read-outs carry less error than TensorFold's bf16 ones; the fp32 state stays within fp16 rounding."""

    q, k, v, gate, beta, state = _rows(300, 1)
    y_ref, s_ref = _reference(q, k, v, gate, beta, state)
    s_tf = torch.empty_like(state)
    y_tf = tf_gdn.chain(q, k, v, gate, beta, state, s_tf)
    y, s = _thor(q, k, v, gate, beta, state)
    assert y.dtype == torch.float16
    assert _rel(y, y_ref) <= 1.5 * _rel(y_tf, y_ref), (_rel(y, y_ref), _rel(y_tf, y_ref))
    assert _rel(s, s_ref) < 2e-3


def test_splits_on_the_grid_keep_every_bit(thor):
    """A chain split on the 64-row grid, the state handed over, and the first rows of a longer call keep their bits."""

    q, k, v, gate, beta, state = _rows(300, 2)
    y, s = _thor(q, k, v, gate, beta, state)
    y2, s2 = _thor(q, k, v, gate, beta, state)
    assert torch.equal(y, y2) and torch.equal(s, s2)                       # the same call, the same bits
    for cut in (64, 128, 192, 256):
        ya, sa = _thor(q[:cut], k[:cut], v[:cut], gate[:cut], beta[:cut], state)
        yb, sb = _thor(q[cut:], k[cut:], v[cut:], gate[cut:], beta[cut:], sa)
        assert torch.equal(torch.cat([ya, yb]), y) and torch.equal(sb, s), cut
    for m in (1, 63, 65, 255, 299):                                        # a partial last chunk
        ym, _ = _thor(q[:m], k[:m], v[:m], gate[:m], beta[:m], state)
        assert torch.equal(ym, y[:m]), m


def test_kept_points_and_pieces_sit_on_the_grid_below_the_limit(monkeypatch):
    assert [thor_gdn.kept_point(n) for n in (1, 2, 64, 65, 66, 128, 129, 300)] == [None] * 3 + [64, 64, 64, 128, 256]
    assert thor_gdn.piece_rows(0, 100, 300) == 64 and thor_gdn.piece_rows(64, 236, 300) == 236
    assert thor_gdn.piece_rows(128, 40, 300) == 0 and thor_gdn.piece_rows(0, 300, 300) == 300
    monkeypatch.setattr(thor_gdn, "LIMIT", 128)                          # past it: TensorFold's own points and pieces
    assert [thor_gdn.kept_point(n) for n in (100, 129, 130, 300)] == [64, 128, 129, 299]
    assert thor_gdn.piece_rows(0, 100, 300) == 64 and thor_gdn.piece_rows(64, 70, 300) == 70
    assert thor_gdn.piece_rows(130, 41, 300) == 41


def test_rows_below_the_limit_run_chunked(monkeypatch):
    monkeypatch.setattr(thor_gdn, "LIMIT", 128)
    assert [thor_gdn.chunked_rows(p, n) for p, n in ((0, 64), (64, 128), (128, 64), (192, 10), (0, 300))] == \
        [64, 64, 0, 0, 128]


def test_the_front_writes_its_values_rounded_once_and_the_back_reads_them(thor):
    """16-bit fronts round TensorFold's values once; 8 rows a block give the same bits; the back reads fp16 and fp32
    read-outs as it reads bf16 ones."""

    rows = 70                                                              # 8 rows a block: a partial last block
    p, cs, cw, state, a_log, dt, nw = _inputs(rows, 3)
    win = _chain_windows(rows)
    q, k, v, g, beta = _front(p, [cs], win, [0] * rows, cw, a_log, dt)
    ptrs = torch.tensor([cs.data_ptr()], dtype=torch.int64, device=DEV)
    sid, wins = torch.zeros(rows, dtype=torch.int32, device=DEV), torch.tensor(win, dtype=torch.int32, device=DEV)
    for dt16 in (torch.float32, torch.float16, torch.bfloat16):
        for rb in (1, 8):
            out = (torch.empty_like(q, dtype=dt16), torch.empty_like(k, dtype=dt16),
                   torch.empty_like(v, dtype=torch.bfloat16 if dt16 == torch.float32 else dt16),
                   torch.empty_like(g), torch.empty_like(beta))
            gdn_io.front(p, ptrs, sid, wins, cw, a_log, dt, gdn.NK, out=out, rows_per_block=rb)
            assert torch.equal(out[0], q.to(dt16)) and torch.equal(out[1], k.to(dt16)), (dt16, rb)
            assert torch.equal(out[2], v.to(out[2].dtype)) and torch.equal(out[3], g), (dt16, rb)
            assert torch.equal(out[4], beta), (dt16, rb)
    y = torch.randn((rows, gdn.NV, gdn.DV), device=DEV) * 0.3
    y = torch.where(y.abs() < 1e-3, torch.full_like(y, 1e-3), y).bfloat16()   # fp16 holds each value exactly
    want, want_xs = torch.empty((rows, gdn.NV * gdn.DV), dtype=torch.bfloat16, device=DEV), \
        torch.empty((rows, gdn.NV * gdn.DV // 32), dtype=torch.float32, device=DEV)
    gdn_io.back(y, p, nw, 1e-6, want, want_xs)
    for yy in (y, y.half(), y.float()):                                    # the same values: the same bits
        for rb in (1, 8):
            got, got_xs = torch.empty_like(want), torch.empty_like(want_xs)
            gdn_io.back(yy, p, nw, 1e-6, got, got_xs, rows_per_block=rb)
            assert torch.equal(got, want) and torch.equal(got_xs, want_xs), (yy.dtype, rb)


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
@pytest.mark.parametrize("sampling", [None, Sampling(seed=11, top_k=20, top_p=0.95)])
def test_prompt_passes_on_the_grid_resumes_and_drafts_keep_the_bits(thor, sampling, kv_dtype):
    """Passes of 64, 128 or 192 rows equal one pass; a resume from a kept point on the grid equals a fresh prefill;
    MTP drafts keep the serial tokens."""

    w = _model()
    prompt = [(37 * i + 11) % V for i in range(300)]
    ref_e = Engine(w, capacity=1024, max_rows=8, prefill_rows=320, graphs=True, kv_dtype=kv_dtype)
    first = prefill(ref_e, prompt, sampling)
    want = _state(ref_e)
    ref = serial_decode(ref_e, first, 20, sampling).tokens
    for rows in (64, 128, 192):
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=rows, graphs=True, kv_dtype=kv_dtype)
        assert prefill(e, prompt, sampling) == first, rows
        assert all(torch.equal(a, b) for a, b in zip(_state(e), want)), rows
        assert mtp_decode(e, first, 20, sampling, depth=4, confidence=0.0).tokens == ref, rows
    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=64, graphs=True, kv_dtype=kv_dtype)
    prefill(e, prompt[:128], sampling)
    kept = {"state": e.st.snapshot(), "tail": e.last_streams.clone()}
    serial_decode(e, 5, 9, sampling)                                       # a reply decodes past the kept prompt
    assert prefill(e, prompt, sampling, resume=kept) == first
    assert all(torch.equal(a, b) for a, b in zip(_state(e), want))
    assert mtp_decode(e, first, 20, sampling, depth=4, confidence=0.0).tokens == ref


@pytest.mark.parametrize("mtp", [True, False])
@pytest.mark.parametrize("sampling", [None, Sampling(seed=13, top_k=20, top_p=0.95)])
def test_the_kept_point_on_the_grid_resumes_the_same_prompt_and_a_next_turn(thor, sampling, mtp):
    w = _model()
    prompt = [(37 * i + 11) % V for i in range(300)]
    turn = prompt[:299] + [271, 77, 78] + [(13 * i + 5) % V for i in range(90)]
    assert entry_end(prompt) == thor_gdn.kept_point(300) and entry_end(prompt[:64]) is None

    def engine():
        return Engine(w, capacity=1024, max_rows=8, prefill_rows=128, graphs=True, kv_dtype="int8")

    def decode(e, first):
        if mtp:
            return mtp_decode(e, first, 20, sampling, depth=4, confidence=0.0).tokens
        return serial_decode(e, first, 20, sampling).tokens

    def fresh(p):
        e = engine()
        first = prefill(e, p, sampling, mtp=mtp)
        return first, _state(e), decode(e, first)

    first, want, ref = fresh(prompt)
    turn_first, turn_want, turn_ref = fresh(turn)
    e = engine()
    assert prefill(e, prompt, sampling, mtp=mtp, keep_at=entry_end(prompt)) == first
    kept = e.kept
    assert kept["state"]["pos"] == entry_end(prompt)                    # 256 on the grid; 299 past a limit of 128
    assert all(torch.equal(a, b) for a, b in zip(_state(e), want)) and decode(e, first) == ref
    assert prefill(e, prompt, sampling, mtp=mtp, resume=kept, keep_at=entry_end(prompt)) == first
    assert e.kept is kept                                                  # the same prompt again keeps its point
    assert all(torch.equal(a, b) for a, b in zip(_state(e), want)) and decode(e, first) == ref
    assert prefill(e, turn, sampling, mtp=mtp, resume=kept, keep_at=entry_end(turn)) == turn_first
    assert all(torch.equal(a, b) for a, b in zip(_state(e), turn_want)) and decode(e, turn_first) == turn_ref


@pytest.mark.parametrize("decoding", [False, True])
def test_packed_passes_end_pieces_on_the_grid_and_keep_each_prompts_solo_run(thor, decoding):
    """Prompts sharing 192-row passes (pieces cut back to the grid), beside a decoding stream or alone, emit their solo
    runs; a resend and a next turn resume from the kept point on the grid."""

    w = _model(5)
    g = torch.Generator().manual_seed(12)
    prompts = [torch.randint(1, V, (n,), generator=g).tolist() for n in (40, 150, 70, 333, 90)]
    samplings = [None, Sampling(seed=8, top_k=20, top_p=0.95), None, Sampling(seed=9, top_k=20, top_p=0.95), None]

    def fresh(prompt, sampling, count=14):
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=64, kv_dtype="int8")
        return serial_decode(e, prefill(e, prompt, sampling), count, sampling).tokens

    dec = MultiDecoder(w, slots=5, capacity=1024, depth=3, confidence=0.3, kv_dtype="int8", prefill_rows=192)
    streams = [Stream(p, 14, smp) for p, smp in zip(prompts, samplings)]
    if decoding:
        dec.admit(streams[0])
        dec.finish(dec.round())
    for s in streams[1 if decoding else 0:]:
        dec.admit(s)
    while dec.live():
        dec.finish(dec.round())
    assert [s.out for s in streams] == [fresh(p, smp) for p, smp in zip(prompts, samplings)]
    assert sorted(len(k[0]) for k in dec.kept) == sorted(entry_end(p) for p in prompts if entry_end(p))
    for p, smp in zip(prompts, samplings):
        for q in (p, p[:-1] + [271, 77]):                                  # the same prompt again, then a next turn
            s = Stream(list(q), 14, smp)
            dec.admit(s)
            while dec.live():
                dec.finish(dec.round())
            assert s.cached == (entry_end(p) or 0) and s.out == fresh(q, smp), (len(p), q[-2:])
