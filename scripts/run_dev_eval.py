"""End-to-end dev eval: pools -> (fake/real) describe -> encode -> retrieve -> recall."""
import sys
import numpy as np
from covrx.config import load_covrx_config
from covrx.pools import load_pool_records
from covrx.encoder import get_encoder
from covrx.index import build_gallery
from covrx.retrieve import rank_pool, recall_from_ranks
from covrx._baseline import list_videos


def main(cfg_path: str):
    cfg = load_covrx_config(cfg_path)
    enc = get_encoder(cfg.encoder, model=cfg.encoder_model,
                      pretrained=cfg.encoder_pretrained, num_frames=cfg.num_frames)

    def loader(path, n):
        return [np.zeros((8, 8, 3), np.uint8)]  # dummy-encoder placeholder

    build_gallery(list_videos(cfg.video_dir), enc, cfg.gallery_npz,
                  frame_loader=loader, num_frames=1)
    gallery = dict(np.load(cfg.gallery_npz))
    pools = load_pool_records(cfg.label_path, cfg.video_dir)

    for pool, recs in pools.items():
        ranked_lists, gts = [], []
        for r in recs:
            q = enc.encode_text(r.edit_instruction)  # describe stub for dummy path
            ranked_lists.append(rank_pool(q, r, gallery))
            gts.append(r.target_key)
        print(pool, recall_from_ranks(ranked_lists, gts))


if __name__ == "__main__":
    main(sys.argv[1])
