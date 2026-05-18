from typing import Callable, Dict, List
import numpy as np

from covrx.fusion import fuse_scores

Scorer = Callable[[str, str], float]


def rerank_topk(description: str, ranked: List[str],
                retrieval_sims: Dict[str, float], scorer: Scorer,
                top_k: int, alpha: float) -> List[str]:
    head = ranked[:top_k]
    tail = ranked[top_k:]
    try:
        rr = np.array([float(scorer(description, k)) for k in head], np.float32)
    except Exception as e:
        print(f"[rerank] fallback (scorer failed): {e}")
        return ranked
    retr = np.array([retrieval_sims[k] for k in head], np.float32)
    fused = fuse_scores(retr, rr, alpha)
    order = np.argsort(-fused)
    return [head[i] for i in order] + tail


def make_qwen_scorer(processor, sample_fps: float, key_to_path: Dict[str, str]) -> Scorer:
    """Pointwise 0-1 relevance of a candidate video to the target description."""
    from covrx._baseline import baseline_evaluate
    from data import VideoDataset, collate_fn
    from torch.utils.data import DataLoader

    def scorer(description: str, key: str) -> float:
        prompt = (
            f'Target description: "{description}"\n\n'
            "Does this video match the target description? "
            "Answer ONLY a number from 0.0 (no match) to 1.0 (perfect match)."
        )
        ds = VideoDataset([key_to_path[key]], prompt, sample_fps)
        loader = DataLoader(ds, batch_size=1, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, processor.processor))
        for inputs, _ in loader:
            txt = baseline_evaluate._generate_text(
                processor, inputs, max_tokens=8,
                temperature=0.0, top_p=1.0, top_k=1, do_sample=False)
            for tok in txt.replace(",", " ").split():
                try:
                    return max(0.0, min(1.0, float(tok)))
                except ValueError:
                    continue
        return 0.0

    return scorer
