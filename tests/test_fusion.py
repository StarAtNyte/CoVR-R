import numpy as np
from covrx.fusion import minmax_norm, fuse_scores


def test_minmax_norm():
    out = minmax_norm(np.array([0., 5., 10.]))
    assert np.allclose(out, [0.0, 0.5, 1.0])
    flat = minmax_norm(np.array([3., 3., 3.]))
    assert np.allclose(flat, [0.0, 0.0, 0.0])


def test_fuse_retrieval_and_rerank():
    retr = np.array([0.0, 1.0])
    rr = np.array([1.0, 0.0])
    fused = fuse_scores(retr, rr, alpha=0.5)
    assert np.isclose(fused[0], fused[1])
    fused_r = fuse_scores(retr, rr, alpha=1.0)
    assert fused_r[1] > fused_r[0]


def test_dual_sim_combine():
    from covrx.fusion import combine_dual_sim
    primary = np.array([0.0, 1.0])
    secondary = np.array([1.0, 0.0])
    c = combine_dual_sim(primary, secondary, weight=0.0)
    assert np.allclose(c, primary)
    c2 = combine_dual_sim(primary, secondary, weight=1.0)
    assert np.allclose(c2, secondary)
