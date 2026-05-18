import numpy as np
from covrx.rerank import rerank_topk


def test_rerank_reorders_topk_only():
    ranked = ["a.mp4", "b.mp4", "c.mp4", "d.mp4"]
    retr_sims = {"a.mp4": 0.9, "b.mp4": 0.8, "c.mp4": 0.7, "d.mp4": 0.1}

    def scorer(desc, key):
        return {"a.mp4": 0.1, "b.mp4": 0.9, "c.mp4": 0.2}[key]

    out = rerank_topk("desc", ranked, retr_sims, scorer,
                       top_k=3, alpha=0.0)  # pure rerank within top-3
    assert out[:1] == ["b.mp4"]
    assert out[-1] == "d.mp4"


def test_rerank_scorer_failure_falls_back():
    ranked = ["a.mp4", "b.mp4", "c.mp4"]
    retr = {"a.mp4": 0.9, "b.mp4": 0.5, "c.mp4": 0.1}

    def bad(desc, key):
        raise RuntimeError("vlm timeout")

    out = rerank_topk("d", ranked, retr, bad, top_k=3, alpha=0.6)
    assert out == ranked
