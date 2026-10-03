"""tools/thor/decode_bytes.py: a decode round's weight bytes from a checkpoint inventory (DECODE-0's inv.py JSON) and
measured MTP statistics (sparkdash_summary.py JSON): T-025's verify-pass model (unique routed experts under independent
top-k routing plus the fixed dense, hc, head and router bytes) extended with the MTP draft steps, against the DRAM floor."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import decode_bytes as db  # noqa: E402

NAMES = {
    # NVFP4 (local-inference-lab) names, digits replaced by '#'
    "model.language_model.layers.#.mlp.experts.#.down_proj.weight": "expert",
    "model.language_model.layers.#.mlp.experts.#.gate_proj.weight_scale": "expert",
    "mtp.layers.#.mlp.experts.#.up_proj.weight": "mtp_expert",
    "mtp.layers.#.self_attn.q_proj.weight": "mtp",
    "mtp.layers.#.mlp.gate.weight": "mtp",
    "mtp.fc_embedding.weight": "mtp",
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight": "mtp",
    "lm_head.weight": "head",
    "model.language_model.embed_tokens.weight": "skip",
    "model.language_model.layers.#.ple.ple_embedding.ngram_embedding.shard_#.weight": "skip",
    "model.language_model.layers.#.ple.key_proj.weight": "dense",
    "model.language_model.layers.#.mlp.gate.weight": "router",
    "model.language_model.layers.#.mlp.shared_expert_gate.weight": "dense",
    "model.language_model.layers.#.mlp.shared_expert.gate_proj.weight": "dense",
    "model.language_model.layers.#.attn_hyper_connection.input_mix_weight_down.weight": "hc",
    "model.language_model.layers.#.mlp_hyper_connection.hc_norm.weight": "hc",
    "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight": "hc",
    "model.language_model.layers.#.linear_attn.in_proj_qkv.weight": "dense",
    "model.language_model.layers.#.self_attn.o_proj.weight": "dense",
    "model.language_model.norm.weight": "skip",
    "model.visual.blocks.#.mlp.linear_fc#.weight": "skip",
    # MLX 4-bit (Vontra / TensorFold) names
    "language_model.model.layers.#.mlp.switch_mlp.down_proj.scales": "expert",
    "language_model.mtp.layers.#.mlp.switch_mlp.gate_proj.biases": "mtp_expert",
    "language_model.mtp.fc_hidden.weight": "mtp",
    "language_model.lm_head.weight": "head",
    "language_model.lm_head.scales": "head",
    "language_model.model.embed_tokens.weight": "skip",
    "language_model.model.layers.#.ple.ple_embedding.ngram_embedding.shard_#.biases": "skip",
    "language_model.model.layers.#.mlp.gate.weight": "router",
    "language_model.model.layers.#.attn_hyper_connection.block_inject_weight.weight": "hc",
    "language_model.model.hyper_connection_mixer.input_mix_weight_up.weight": "hc",
    "language_model.model.layers.#.linear_attn.in_proj_a.weight": "dense",
    "vision_tower.blocks.#.attn.qkv.weight": "skip",
}


@pytest.mark.parametrize("name,cls", sorted(NAMES.items()))
def test_classify(name, cls):
    assert db.classify(name) == cls


def inv(groups, config=None):
    return {"snapshot": "x", "files": 1, "total_bytes": sum(g[3] for g in groups), "groups": groups,
            "config": config if config is not None else {}}


def test_classes_sums_every_class():
    i = inv([["model.language_model.layers.#.mlp.experts.#.down_proj.weight", "U8", 4, 400, [2, 2]],
             ["model.language_model.layers.#.mlp.experts.#.up_proj.weight", "U8", 4, 600, [2, 2]],
             ["lm_head.weight", "BF16", 1, 50, [2, 2]],
             ["model.visual.blocks.#.attn.qkv.weight", "F8_E4M3", 2, 7, [2, 2]]])
    c = db.classes(i)
    assert set(c) == {"expert", "mtp_expert", "router", "hc", "head", "mtp", "dense", "skip"}
    assert c["expert"] == 1000 and c["head"] == 50 and c["skip"] == 7 and c["dense"] == 0 and c["mtp"] == 0


def test_model_dims_from_text_config_and_defaults():
    cfg = {"text_config": {"num_hidden_layers": 4, "num_experts": 16, "num_experts_per_tok": 3, "hidden_size": 64,
                           "vocab_size": 1000}}
    assert db.model_dims(inv([], cfg)) == {"layers": 4, "experts": 16, "topk": 3, "hidden": 64, "vocab": 1000}
    top = {"num_hidden_layers": 6, "num_experts": 32, "num_experts_per_tok": 4, "hidden_size": 128, "vocab_size": 77}
    assert db.model_dims(inv([], top)) == {"layers": 6, "experts": 32, "topk": 4, "hidden": 128, "vocab": 77}
    assert db.model_dims(inv([], {})) == {"layers": 48, "experts": 512, "topk": 10, "hidden": 2560, "vocab": 248320}


def test_draft_head_bytes():
    assert db.draft_head_bytes(79591, 2560) == pytest.approx(79591 * 2560 * 0.625)
    assert db.draft_head_bytes(100, 64, bits=8, group=64) == pytest.approx(100 * 64 * (1 + 4 / 64))


CLS = {"expert": 2 * 8 * 1e6, "mtp_expert": 8 * 2e6, "router": 1e8, "hc": 2e8, "head": 3e8, "mtp": 5e7, "dense": 1e9,
       "skip": 123.0}
DIMS = {"layers": 2, "experts": 8, "topk": 2, "hidden": 64, "vocab": 1000}


def test_round_bytes_hand_computed():
    r = db.round_bytes(CLS, DIMS, tpr=3.0, acc=0.5, users=1, draft_head=1e7, bw_gbps=256.0)
    assert r["drafts"] == pytest.approx(4.0)
    assert r["rows"] == pytest.approx(5.0)
    unique = 8 * (1 - 0.75 ** 5)
    assert r["unique_experts"] == pytest.approx(unique)
    assert r["expert_gb"] == pytest.approx(unique * 2 * 1e6 / 1e9)
    assert r["fixed_gb"] == pytest.approx(1.6)
    assert r["verify_gb"] == pytest.approx(1.6 + unique * 2e6 / 1e9)
    assert r["draft_step_gb"] == pytest.approx((5e7 + 2 * 2e6 + 1e7) / 1e9)
    assert r["draft_gb"] == pytest.approx(4 * 6.4e7 / 1e9)
    total = 1.6e9 + unique * 2e6 + 4 * 6.4e7
    assert r["total_gb"] == pytest.approx(total / 1e9)
    assert r["floor_ms"] == pytest.approx(total / 256e9 * 1e3)
    assert r["verify_floor_ms"] == pytest.approx((1.6e9 + unique * 2e6) / 256e9 * 1e3)


def test_round_bytes_users_and_no_drafts():
    r = db.round_bytes(CLS, DIMS, tpr=2.0, acc=0.5, users=4, draft_head=0.0)
    assert r["drafts"] == pytest.approx(2.0) and r["rows"] == pytest.approx(12.0)
    assert r["unique_experts"] == pytest.approx(8 * (1 - 0.75 ** 12))
    assert r["draft_step_gb"] == pytest.approx((5e7 + 8 * (1 - 0.75 ** 4) * 2e6) / 1e9)
    assert r["floor_ms"] == pytest.approx(r["total_gb"] / 256.0 * 1e3)   # default bandwidth 256 GB/s (T-025.o18)
    z = db.round_bytes(CLS, DIMS, tpr=1.0, acc=0.0, users=1, draft_head=0.0)
    assert z["drafts"] == 0.0 and z["rows"] == pytest.approx(1.0) and z["draft_gb"] == 0.0
    n = db.round_bytes(CLS, DIMS, tpr=2.5, acc=None, users=1, draft_head=0.0)
    assert n["drafts"] == 0.0


def summary():
    def row(config, kind, conc, tpr, acc, per):
        return {"config": config, "type": kind, "conc": conc, "n": 5, "failed": 0,
                "agg": {"n": 5, "mean": per * conc}, "per_req": {"n": 5, "mean": per},
                "acceptance": acc, "tokens_per_round": tpr}
    return {"decode": [row("A", "prose", 1, 3.0, 0.5, 50.0), row("B", "prose", 1, 3.0, 0.5, 60.0),
                       row("B", "code", 4, None, None, 30.0), row("Z", "prose", 1, 3.0, 0.5, 10.0)],
            "prefill": []}


def test_table_rows_and_measured_round():
    invs = {"A": inv([["layers.#.mlp.experts.#.w", "U8", 1, 2 * 8 * 1e6, [1]], ["lm_head.weight", "BF16", 1, 3e8, [1]]],
                     {"text_config": DIMS_CFG}),
            "B": inv([["layers.#.mlp.experts.#.w", "U8", 1, 2 * 8 * 2e6, [1]]], {"text_config": DIMS_CFG})}
    rows = db.table(summary(), invs, draft_ids=10)
    assert [(r["config"], r["type"], r["conc"]) for r in rows] == [("A", "prose", 1), ("B", "prose", 1)]
    a = rows[0]
    assert a["tpr"] == 3.0 and a["acc"] == 0.5
    assert a["measured_ms"] == pytest.approx(1000 * 3.0 / 50.0)
    want = db.round_bytes(db.classes(invs["A"]), db.model_dims(invs["A"]), 3.0, 0.5, users=1,
                          draft_head=db.draft_head_bytes(10, 64))
    assert a["total_gb"] == pytest.approx(want["total_gb"]) and a["floor_ms"] == pytest.approx(want["floor_ms"])
    assert a["floor_pct"] == pytest.approx(100 * want["floor_ms"] / a["measured_ms"])


DIMS_CFG = {"num_hidden_layers": 2, "num_experts": 8, "num_experts_per_tok": 2, "hidden_size": 64, "vocab_size": 1000}


def test_markdown_header_and_cells():
    invs = {"A": inv([["layers.#.mlp.experts.#.w", "U8", 1, 2 * 8 * 1e6, [1]]], {"text_config": DIMS_CFG})}
    rows = db.table(summary(), invs, draft_ids=0)
    md = db.markdown(rows)
    lines = md.splitlines()
    assert lines[0] == ("| config | type | streams | tokens/round | acceptance | drafts | verify rows | unique experts "
                        "| expert GB | fixed GB | draft GB | total GB | floor ms | measured ms/round | floor % |")
    assert lines[1] == "|---" * 15 + "|"
    assert lines[2].startswith("| A | prose | 1 | 3.00 | 0.500 | 4.00 | 5.00 | ")
    assert lines[2].endswith(" | 60.0 | " + f"{rows[0]['floor_pct']:.0f}%" + " |")
    assert md.endswith("\n") and len(lines) == 3


def test_main_end_to_end(tmp_path, capsys):
    s = tmp_path / "summary.json"
    s.write_text(json.dumps(summary()))
    ia = tmp_path / "inv_a.json"
    ia.write_text(json.dumps(inv([["layers.#.mlp.experts.#.w", "U8", 1, 2 * 8 * 1e6, [1]]], {"text_config": DIMS_CFG})))
    out_j, out_m = tmp_path / "bytes.json", tmp_path / "bytes.md"
    rc = db.main([str(s), "--inv", f"A={ia}", "--draft-ids", "0", "--bw", "128", "--json", str(out_j),
                  "--markdown", str(out_m)])
    assert rc == 0
    got = json.loads(out_j.read_text())
    assert got["bw_gbps"] == 128.0 and got["draft_ids"] == 0
    assert [r["config"] for r in got["rows"]] == ["A"]
    assert got["rows"][0]["floor_ms"] == pytest.approx(got["rows"][0]["total_gb"] / 128.0 * 1e3)
    assert "classes" in got and got["classes"]["A"]["expert"] == 2 * 8 * 1e6
    assert out_m.read_text() == capsys.readouterr().out
