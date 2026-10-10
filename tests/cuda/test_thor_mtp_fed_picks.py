"""Thor's device-fed MTP chain picks kernel (DPUSH-6, ``thor_mtp_fed.picks``): each row's top logit and the first
column holding it (``torch.max``'s), its draft id, and a logsumexp within float32 rounding of torch's; a row's bits are
the same whatever rows share the launch (``torch.logsumexp``'s are not: its reduction order follows the row count), and
a CUDA graph's replay gives the eager bits."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.qwen4_exp.cuda import thor_mtp_fed  # noqa: E402

thor = pytest.mark.skipif(torch.cuda.get_device_capability() != (11, 0), reason="Thor (sm_110) only")


def _logits(rows: int, n: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn((rows, n), generator=gen, device="cuda") * 3).bfloat16()
    for r in range(rows):                        # ties at the top: the first column must win
        top = x[r].float().max()
        cols = torch.randint(0, n, (3,), generator=gen, device="cuda")
        x[r, cols] = (top + 1).bfloat16()
    return x


def _run(x: torch.Tensor, idmap):
    slots = max(8, x.shape[0])
    got = torch.full((slots, 3), float("nan"), device="cuda")
    tok = torch.full((slots,), -1, dtype=torch.int32, device="cuda")
    thor_mtp_fed.picks(x, got, tok, idmap)
    torch.cuda.synchronize()
    return got[:x.shape[0]].clone(), tok[:x.shape[0]].clone()


@thor
@pytest.mark.parametrize("n", [32768, 151936, 5003])
def test_picks_match_torch_and_keep_their_bits_in_any_batch(n):
    x = _logits(8, n, seed=n)
    idmap = torch.randperm(n, device="cuda").to(torch.int32) + 7
    got, tok = _run(x, idmap)
    ref = x.float()
    top, col = ref.max(dim=-1)
    assert torch.equal(got[:, 0], top)
    assert torch.equal(got[:, 1].long(), col)
    assert torch.equal(tok, idmap[col])
    torch.testing.assert_close(got[:, 2], torch.logsumexp(ref, dim=-1), rtol=2e-6, atol=2e-6)
    for rows in range(1, 9):                      # the same rows alone, in fewer-row and offset batches
        sub, subtok = _run(x[:rows], idmap)
        assert torch.equal(sub.view(torch.int32), got[:rows].view(torch.int32)) and torch.equal(subtok, tok[:rows])
        one, onetok = _run(x[rows - 1:rows], None)
        assert torch.equal(one.view(torch.int32), got[rows - 1:rows].view(torch.int32))
        assert torch.equal(onetok.long(), col[rows - 1:rows])


@thor
def test_picks_replay_in_a_graph():
    x = _logits(5, 32768, seed=3)
    want, wtok = _run(x, None)
    got = torch.zeros((8, 3), device="cuda")
    tok = torch.zeros((8,), dtype=torch.int32, device="cuda")
    static = x.clone()
    thor_mtp_fed.picks(static, got, tok, None)           # compiled before the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        thor_mtp_fed.picks(static, got, tok, None)
    got.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(got[:5].view(torch.int32), want.view(torch.int32)) and torch.equal(tok[:5], wtok)
