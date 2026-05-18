import os
from typing import Dict, List, Tuple
import numpy as np

from covrx.pools import PoolRecord


def rank_pool(query_vec: np.ndarray, rec: PoolRecord,
              gallery: Dict[str, np.ndarray]) -> List[str]:
    keys = sorted(gallery)
    mat = np.stack([gallery[k] for k in keys]).astype(np.float32)
    sims = mat @ query_vec.astype(np.float32)
    if rec.reference_key in gallery:
        sims[keys.index(rec.reference_key)] = -np.inf
    order = np.argsort(-sims)
    return [keys[i] for i in order]


def recall_from_ranks(ranked_lists: List[List[str]], gt_keys: List[str]) -> Dict[str, float]:
    ranks = []
    for ranked, gt in zip(ranked_lists, gt_keys):
        ranks.append(ranked.index(gt) if gt in ranked else 10**9)
    ranks = np.asarray(ranks)
    n = len(ranks)
    f = lambda k: round(100.0 * np.count_nonzero(ranks < k) / n, 4)
    return {"R1": f(1), "R5": f(5), "R10": f(10), "R50": f(50)}


def _token(key: str) -> str:
    return os.path.splitext(key)[0]


def build_submission(
    per_pool: Dict[str, List[Tuple[PoolRecord, List[str]]]],
    top_k: int = 50,
) -> Dict[str, list]:
    out: Dict[str, list] = {}
    for pool, items in per_pool.items():
        out[pool] = [
            {"id": rec.index, "ranking": [_token(k) for k in ranked[:top_k]]}
            for rec, ranked in items
        ]
    return out
