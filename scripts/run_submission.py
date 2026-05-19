"""Full GPU pipeline → submission JSON. Run via Modal for val/test."""
import json
import os
import sys
import numpy as np

from covrx.config import load_covrx_config
from covrx.pools import load_pool_records
from covrx.encoder import get_encoder
from covrx.frames import sample_frames
from covrx.index import build_gallery, pool_gallery_path
from covrx.retrieve import rank_pool, build_submission
from covrx.rerank import rerank_topk
from covrx._baseline import list_videos


def assemble_submission(per_pool, out_path, top_k):
    sub = build_submission(per_pool, top_k=top_k)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(sub, f)
    return sub


def run(cfg_path: str):
    cfg = load_covrx_config(cfg_path)
    enc = get_encoder(cfg.encoder, model=cfg.encoder_model,
                      pretrained=cfg.encoder_pretrained, num_frames=cfg.num_frames)

    # 2. reasoning backend (Qwen3-VL via baseline VideoProcessor) — created once
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                    "CoVR-R-authors-github"))
    from config import Config
    from model import VideoProcessor
    from covrx.reason import ReasonedDescriber, make_qwen_generate
    from covrx.rerank import make_qwen_scorer
    bcfg = Config(model_name=cfg.vlm_model, parallelism="mp")
    proc = VideoProcessor(bcfg)
    proc.load_model(0)
    describer = ReasonedDescriber(make_qwen_generate(proc, cfg.sample_fps),
                                  cfg.n_self_consistency)

    pools = load_pool_records(cfg.label_path, cfg.video_dir, cfg.pool_subdir)
    per_pool = {}
    for pool, recs in pools.items():
        # 1. Build and load gallery ONLY for this pool
        pool_dir = os.path.join(cfg.video_dir, cfg.pool_subdir.get(pool, pool))
        gpath = pool_gallery_path(cfg.gallery_npz, pool)
        build_gallery(list_videos(pool_dir), enc, gpath,
                      frame_loader=sample_frames, num_frames=cfg.num_frames)
        gallery = dict(np.load(gpath))
        key_to_path = {os.path.basename(p): p for p in list_videos(pool_dir)}

        scorer = (make_qwen_scorer(proc, cfg.sample_fps, key_to_path)
                  if cfg.rerank_enabled else None)

        items = []
        for r in recs:
            desc = describer.describe(r.video_path, r.edit_instruction).target_description
            q = enc.encode_text(desc)
            ranked = rank_pool(q, r, gallery)
            if scorer is not None:
                keys = sorted(gallery)
                mat = np.stack([gallery[k] for k in keys])
                sims = {k: float(s) for k, s in zip(keys, mat @ q)}
                ranked = rerank_topk(desc, ranked, sims, scorer,
                                     cfg.top_k, cfg.fusion_alpha)
            items.append((r, ranked))
        per_pool[pool] = items
    assemble_submission(per_pool, cfg.submission_path, cfg.top_k)
    print("wrote", cfg.submission_path)


if __name__ == "__main__":
    run(sys.argv[1])
