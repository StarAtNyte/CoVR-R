import numpy as np
from covrx.retrieve import rank_pool, recall_from_ranks, build_submission
from covrx.pools import PoolRecord


def _rec(idx, ref, tgt, ext=".mp4"):
    return PoolRecord(idx, "webvid", ref, tgt, "edit",
                      f"./videos/webvid/{ref}{ext}", ext)


def test_rank_masks_reference_and_orders():
    gallery = {"a.mp4": np.array([1., 0.], np.float32),
               "b.mp4": np.array([0., 1.], np.float32),
               "c.mp4": np.array([0.7, 0.7], np.float32)}
    q = np.array([1., 0.], np.float32)
    rec = _rec(0, "a", "c")
    ranked = rank_pool(q, rec, gallery)
    assert ranked[0] != "a.mp4"
    assert ranked[0] == "c.mp4"


def test_recall_from_ranks():
    ranked_lists = [["c.mp4", "b.mp4"], ["x.mp4", "y.mp4", "z.mp4"]]
    gt = ["c.mp4", "z.mp4"]
    m = recall_from_ranks(ranked_lists, gt)
    assert m["R1"] == 50.0
    assert m["R5"] == 100.0


def test_build_submission_shape():
    sub = build_submission({"webvid": [(_rec(0, "a", ""), ["b.mp4", "c.mp4"])]})
    assert sub["webvid"][0]["id"] == 0
    assert sub["webvid"][0]["ranking"] == ["b", "c"]


def test_matches_baseline_compute_recall():
    from covrx._baseline import compute_recall
    scores = np.array([[0.9, 0.1, 0.2], [0.1, 0.2, 0.9]], np.float32)
    base = compute_recall(scores, [0, 2])
    ranked = [["g0", "g2", "g1"], ["g2", "g1", "g0"]]
    ours = recall_from_ranks(ranked, ["g0", "g2"])
    assert ours["R1"] == base["R1"]
