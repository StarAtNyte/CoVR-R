import textwrap
from covrx.config import CovrxConfig, load_covrx_config


def test_defaults():
    c = CovrxConfig()
    assert c.vlm_model == "Qwen/Qwen3-VL-32B-Instruct"
    assert c.encoder == "openclip"
    assert c.rerank_enabled is True
    assert c.dual_sim_enabled is False
    assert c.top_k == 50
    assert 0.0 <= c.fusion_alpha <= 1.0


def test_yaml_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(textwrap.dedent("""
        encoder: dummy
        rerank_enabled: false
        fusion_alpha: 0.7
        n_self_consistency: 3
    """))
    c = load_covrx_config(str(p))
    assert c.encoder == "dummy"
    assert c.rerank_enabled is False
    assert c.fusion_alpha == 0.7
    assert c.n_self_consistency == 3
    assert c.vlm_model == "Qwen/Qwen3-VL-32B-Instruct"  # untouched default
