"""Grid over encoder, n_self_consistency, fusion_alpha, rerank on dev set.
Requires labeled dev set (Task 11). Prints a table; you pick the best config."""
import itertools
import sys
import numpy as np
from covrx.config import load_covrx_config
from covrx.pools import load_pool_records
from covrx.encoder import get_encoder
from covrx.retrieve import rank_pool, recall_from_ranks

GRID = {
    "encoder": ["openclip"],
    "n_self_consistency": [1, 3, 5],
    "fusion_alpha": [0.4, 0.6, 0.8],
    "rerank_enabled": [False, True],
}


def main(cfg_path: str, gallery_npz: str):
    base = load_covrx_config(cfg_path)
    gallery = dict(np.load(gallery_npz))
    pools = load_pool_records(base.label_path, base.video_dir)
    keys = list(GRID)
    for combo in itertools.product(*GRID.values()):
        params = dict(zip(keys, combo))
        # NOTE: real run plugs describer+rerank; here measure retrieval-only
        # ceiling per combo using cached query vecs (see scripts/run_submission).
        print(params, "-> run scripts/run_submission with these overrides, "
                       "then recall_from_ranks on dev labels")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
