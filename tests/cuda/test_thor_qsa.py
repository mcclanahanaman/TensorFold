"""Jetson Thor's sparse-attention prompt kernels (THOR-K4-QSA): the 8-bit cache's bf16 shadow holds the values
TensorFold's kernel dequantizes, the prompt attention on it and the tensor-core block scores agree with TensorFold's
kernels up to rounding (the scores select the same blocks), and Flash Next's prompts keep their bits with the path
on: chunkings, a resume and drafted tokens, past the indexer budget, on an 8-bit and a bf16 cache; dense rows (up to
2,051 keys) are TensorFold's bits exactly."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.qwen4_exp.cuda import thor_qsa  # noqa: E402

if not thor_qsa.available():
    pytest.skip("Jetson Thor (sm_110) with the Thor QSA path on only", allow_module_level=True)

from test_flashnext_forward import V, _model, _state  # noqa: E402

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import attention as attn_mod  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.kvcache import dequant_ref, h32_ref, quantize_ref  # noqa: E402

DEV = "cuda"


@pytest.fixture
def thor_on():
    """The path on for an 8-bit cache, and off again after the test."""

    yield lambda bits: thor_qsa.prepare(bits)
    thor_qsa.set_active(False)


def _kv(cap: int, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    k = (torch.randn((cap, 2, 256), generator=g, device=DEV) * 0.6).to(torch.bfloat16)
    v = (torch.randn((cap, 2, 256), generator=g, device=DEV) * 0.6).to(torch.bfloat16)
    return k, v


def test_the_shadow_holds_the_blocks_the_rows_read_as_the_kernel_dequantizes():
    """``stage`` writes the 4-key blocks the rows read (their selected blocks and tails), each with the values
    TensorFold's kernel dequantizes, and leaves the rest of the shadow alone; ``stage_all`` writes every row."""

    g = torch.Generator(device=DEV).manual_seed(4)
    rows, cap = 4, 12001
    k, v = _kv(cap, 3)
    kc, ks, vc, vs = quantize_ref(k, v, bits=8)
    iq = torch.randn((rows, 4, 128), generator=g, device=DEV).to(torch.bfloat16)
    pooled = torch.randn((4096, 128), generator=g, device=DEV).to(torch.bfloat16)
    pos0 = torch.tensor([cap - rows], dtype=torch.int32, device=DEV)
    scratch = attn_mod.AttnScratch(rows, 24, 256, 16384, DEV, budget=2048, ratio=4)
    attn_mod.qsa_rows(iq, pooled, pos0, scratch, rows, context=cap)
    cache = type("C", (), {"k": kc, "v": vc, "ks": ks, "vs": vs})()
    shadow = thor_qsa.Shadow(16384, 2, 256, DEV, 4)
    shadow.k.fill_(float("nan"))
    shadow.v.fill_(float("nan"))
    thor_qsa.begin(shadow)
    thor_qsa.stage(cache, shadow, scratch, pos0, rows, cap)
    read = torch.zeros((cap,), dtype=torch.bool, device=DEV)
    for r in range(rows):
        read[scratch.ids[r, :int(scratch.nk[r])].long()] = True
    assert int(scratch.sparse.sum()) == rows and 0 < int(read.sum()) < cap
    want_k, want_v = dequant_ref(kc, ks, bits=8), dequant_ref(vc, vs, bits=8)
    assert torch.equal(shadow.k[:cap][read], want_k[read]) and torch.equal(shadow.v[:cap][read], want_v[read])
    blocks = torch.zeros((cap // 4 + 1,), dtype=torch.bool, device=DEV)
    blocks[torch.nonzero(read).squeeze(1) // 4] = True
    staged = blocks.repeat_interleave(4)[:cap]
    assert torch.isnan(shadow.k[:cap][~staged].float()).all()                 # blocks no row reads: left alone
    thor_qsa.stage_all(cache, shadow, cap)
    assert torch.equal(shadow.k[:cap], want_k) and torch.equal(shadow.v[:cap], want_v)
    assert torch.isnan(shadow.k[cap:].float()).all()


@pytest.mark.parametrize("cap,rows", [(600, 64), (3000, 256)], ids=["dense", "sparse"])
def test_prompt_attention_agrees_with_tensorfolds_kernel_to_rounding(cap, rows):
    """Dense rows (below the indexer budget) and sparse rows (their selected blocks): the shadow path's outputs
    against TensorFold's kernel on the 8-bit cache, and each row the same alone as in its block."""

    g = torch.Generator(device=DEV).manual_seed(11)
    heads, d = 24, 256
    k, v = _kv(cap, 12)
    kc, ks, vc, vs = quantize_ref(k, v, bits=8)
    q = (torch.randn((rows, heads, d), generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    qr = h32_ref(q.float().reshape(-1, 32)).to(torch.bfloat16).view(rows, heads, d)
    pos0 = torch.tensor([cap - rows], dtype=torch.int32, device=DEV)
    scratch = attn_mod.AttnScratch(rows, heads, d, 4096, DEV, budget=2048, ratio=4)
    if scratch.qsa:                                               # selections from random block scores
        iq = (torch.randn((rows, 4, 128), generator=g, device=DEV)).to(torch.bfloat16)
        pooled = (torch.randn((1024, 128), generator=g, device=DEV)).to(torch.bfloat16)
        attn_mod.qsa_rows(iq, pooled, pos0, scratch, rows, context=cap)
    ref = attn_mod.attention(qr, kc, vc, pos0, scratch, rows, d ** -0.5, ks=ks, vs=vs, bits=8,
                             context=cap).clone()
    cache = type("C", (), {"k": kc, "v": vc, "ks": ks, "vs": vs})()
    shadow = thor_qsa.Shadow(4096, 2, d, DEV, 4)
    thor_qsa.begin(shadow)
    thor_qsa.stage(cache, shadow, scratch, pos0, rows, cap)
    got = torch.empty_like(ref)
    thor_qsa.attention(qr, cache, shadow, pos0, scratch, rows, d ** -0.5, got, context=cap, bits=8)
    rel = float((got.float() - ref.float()).abs().max()) / float(ref.float().abs().max())
    assert rel < 1e-2, rel
    assert float((got == ref).float().mean()) > 0.99
    for r in (0, rows // 2, rows - 1):                            # a row alone: the same bits as in its block
        one = torch.empty_like(ref[:1])
        sub = attn_mod.AttnScratch(1, heads, d, 4096, DEV, budget=2048, ratio=4)
        if sub.qsa:
            sub.ids[0].copy_(scratch.ids[r])
            sub.nk[0], sub.sparse[0] = scratch.nk[r], scratch.sparse[r]
        thor_qsa.attention(qr[r:r + 1], cache, shadow, pos0 + r, sub, 1, d ** -0.5, one, context=cap - rows + r + 1,
                           bits=8)
        assert torch.equal(one[0], got[r]), r


def test_tensor_core_scores_select_tensorfolds_blocks():
    g = torch.Generator(device=DEV).manual_seed(5)
    rows, cap = 256, 12000
    iq = torch.randn((rows, 4, 128), generator=g, device=DEV).to(torch.bfloat16)
    pooled = torch.randn((cap // 4 + 1, 128), generator=g, device=DEV).to(torch.bfloat16)
    pos0 = torch.tensor([cap - rows], dtype=torch.int32, device=DEV)
    a = attn_mod.AttnScratch(rows, 24, 256, 16384, DEV, budget=2048, ratio=4)
    b = attn_mod.AttnScratch(rows, 24, 256, 16384, DEV, budget=2048, ratio=4)
    attn_mod.qsa_rows(iq, pooled, pos0, a, rows, context=cap)
    thor_qsa.qsa_rows(iq, pooled, pos0, b, rows, context=cap)
    blocks = cap // 4
    complete = (cap - rows + torch.arange(rows, device=DEV) + 1) // 4
    live = torch.arange(blocks, device=DEV)[None, :] < complete[:, None]
    sa, sb = a.scores[:, :blocks], b.scores[:, :blocks]
    assert bool(((sb - sa).abs() <= 1e-5 * (1.0 + sa.abs()))[live].all())     # fp32 sums in another order
    assert torch.equal(a.sparse, b.sparse) and torch.equal(a.nk, b.nk)
    assert torch.equal(a.ids[:, :a.idw], b.ids[:, :b.idw])


@pytest.mark.parametrize("dtype,bits", [("int8", 8), ("bf16", 16)])
@pytest.mark.parametrize("sampling", [None, Sampling(seed=5, top_k=20, top_p=0.95)])
def test_sparse_prompts_keep_their_bits_on_this_path(thor_on, dtype, bits, sampling):
    """TensorFold's sparse-prompt contract with the path on: prompt chunkings and a resume leave the same state,
    drafted tokens equal serial ones; and the first token and state are TensorFold's up to rounding."""

    w = _model()
    prompt = [(37 * i + 11) % V for i in range(2600)]            # rows past 2,051 keys are sparse
    tf_e = Engine(w, capacity=4096, max_rows=8, prefill_rows=2048, kv_dtype=dtype)
    tf_first = prefill(tf_e, prompt, sampling)
    tf_state = _state(tf_e)
    assert thor_on(bits) and thor_qsa.use()
    ref_e = Engine(w, capacity=4096, max_rows=8, prefill_rows=2048, graphs=True, kv_dtype=dtype)
    assert (ref_e.pbuf.thor_shadow is not None) == (bits == 8)
    first = prefill(ref_e, prompt, sampling)
    want = _state(ref_e)
    ref = serial_decode(ref_e, first, 24, sampling).tokens
    for rows, graphs in ((512, True), (1000, False)):
        e = Engine(w, capacity=4096, max_rows=8, prefill_rows=rows, graphs=graphs, kv_dtype=dtype)
        assert prefill(e, prompt, sampling) == first, rows
        assert all(torch.equal(a, b) for a, b in zip(_state(e), want)), rows
        for depth in (3, 6):
            prefill(e, prompt, sampling)
            assert mtp_decode(e, first, 24, sampling, depth=depth, confidence=0.3).tokens == ref, (rows, depth)
    e = Engine(w, capacity=4096, max_rows=8, prefill_rows=512, graphs=True, kv_dtype=dtype)
    prefill(e, prompt[:2300], sampling)
    kept = {"state": e.st.snapshot(), "tail": e.last_streams.clone()}
    serial_decode(e, 5, 9, sampling)                              # a reply decodes past the kept prompt
    assert prefill(e, prompt, sampling, resume=kept) == first
    assert all(torch.equal(a, b) for a, b in zip(_state(e), want))
    assert first == tf_first
    for a, b in zip(want, tf_state):                              # rounding-level differences, if any
        if a.is_floating_point():
            assert float((a.float() - b.float()).abs().max()) <= 0.05 * max(1.0, float(b.float().abs().max()))


def test_a_4bit_cache_keeps_tensorfolds_kernels_after_an_8bit_engine_turned_the_path_on(thor_on):
    """The path is process-wide: once an 8-bit engine has turned it on (its buffers with a shadow), a 4-bit cache's
    prompt passes still run TensorFold's kernels (the shadow stages 8-bit codes only), with TensorFold's bits."""

    w = _model()
    prompt = [(37 * i + 11) % V for i in range(2600)]            # rows past 2,051 keys are sparse
    assert thor_on(8) and thor_qsa.use()
    e = Engine(w, capacity=4096, max_rows=8, prefill_rows=512, kv_dtype="int4")
    assert e.pbuf.thor_shadow is not None                         # buffers built while the 8-bit path was on
    first = prefill(e, prompt, None)
    got = _state(e)
    thor_qsa.set_active(False)
    tf_e = Engine(w, capacity=4096, max_rows=8, prefill_rows=512, kv_dtype="int4")
    assert prefill(tf_e, prompt, None) == first
    assert all(torch.equal(a, b) for a, b in zip(got, _state(tf_e)))


@pytest.mark.parametrize("dtype,bits", [("int8", 8), ("bf16", 16)])
def test_dense_rows_keep_tensorfolds_bits_with_the_path_on(thor_on, dtype, bits):
    """Dense rows keep TensorFold's kernels: a prompt under 2,052 tokens has TensorFold's bits exactly with the path on,
    in one pass or several, and so do its MTP drafts; a longer prompt's first 2,051 rows (the row blocks that hold dense
    and sparse rows split between the two) leave TensorFold's cache bits, whatever the chunking."""

    from test_flashnext_forward import _rows

    w = _model()
    prompt = [(37 * i + 11) % V for i in range(2000)]             # every row dense (at most 2,051 keys)
    tf_e = Engine(w, capacity=4096, max_rows=8, prefill_rows=2048, kv_dtype=dtype)
    tf_first = prefill(tf_e, prompt, None)
    want = _state(tf_e)
    ref = serial_decode(tf_e, tf_first, 24, None).tokens
    longer = [(37 * i + 11) % V for i in range(2600)]
    tf_long = Engine(w, capacity=4096, max_rows=8, prefill_rows=2048, kv_dtype=dtype)
    prefill(tf_long, longer, None)
    assert thor_on(bits) and thor_qsa.use() and thor_qsa.DENSE_ON_TF
    head = thor_qsa.first_sparse(tf_long.pbuf.attn)
    assert head == 2051

    def cache_head(e):
        return [x for c in e.st.kc for x in _rows(c, head)] + [k[:head] for k in e.st.ikc]

    for rows in (2048, 512, 1000):
        e = Engine(w, capacity=4096, max_rows=8, prefill_rows=rows, graphs=True, kv_dtype=dtype)
        assert prefill(e, prompt, None) == tf_first, rows
        assert all(torch.equal(a, b) for a, b in zip(_state(e), want)), rows
        assert mtp_decode(e, tf_first, 24, None, depth=4, confidence=0.3).tokens == ref, rows
        e = Engine(w, capacity=4096, max_rows=8, prefill_rows=rows, graphs=True, kv_dtype=dtype)
        prefill(e, longer, None)
        assert all(torch.equal(a, b) for a, b in zip(cache_head(e), cache_head(tf_long))), rows


def test_whole_and_block_staging_give_the_same_bits():
    """A segment stages its whole context or block by block; the attention reads the same values either way."""

    g = torch.Generator(device=DEV).manual_seed(8)
    rows, cap, heads, d = 256, 9000, 24, 256
    k, v = _kv(cap, 9)
    kc, ks, vc, vs = quantize_ref(k, v, bits=8)
    q = (torch.randn((rows, heads, d), generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    iq = torch.randn((rows, 4, 128), generator=g, device=DEV).to(torch.bfloat16)
    pooled = torch.randn((4096, 128), generator=g, device=DEV).to(torch.bfloat16)
    pos0 = torch.tensor([cap - rows], dtype=torch.int32, device=DEV)
    scratch = attn_mod.AttnScratch(rows, heads, d, 16384, DEV, budget=2048, ratio=4)
    thor_qsa.qsa_rows(iq, pooled, pos0, scratch, rows, context=cap)
    cache = type("C", (), {"k": kc, "v": vc, "ks": ks, "vs": vs})()
    outs = []
    for whole in (True, False):
        shadow = thor_qsa.Shadow(16384, 2, d, DEV, 4)
        if whole:
            thor_qsa.stage_all(cache, shadow, cap)
        else:
            thor_qsa.begin(shadow)
            thor_qsa.stage(cache, shadow, scratch, pos0, rows, cap)
        out = torch.empty((rows, heads, d), dtype=torch.bfloat16, device=DEV)
        thor_qsa.attention(q, cache, shadow, pos0, scratch, rows, d ** -0.5, out, context=cap, bits=8)
        outs.append(out)
    assert torch.equal(outs[0], outs[1])
