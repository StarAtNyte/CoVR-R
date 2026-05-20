"""Full GPU pipeline → submission JSON. Run via Modal for val/test.

Resumable + shardable:
- Each worker writes per-record results to a shard JSONL on the Volume.
- A run can resume mid-shard (skips already-checkpointed records).
- N workers run in parallel via Modal `.map` (sharded by record index).
- `merge_shards` + `assemble_from_done` build the final submission JSON.
"""
import glob
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

CHECKPOINT_EVERY = 10


def shard_path(artifact_dir: str, shard_id: int) -> str:
    return os.path.join(artifact_dir, f"progress.shard_{shard_id}.jsonl")


def load_progress(path: str) -> dict:
    """Return {(pool, id): ranking} from an existing checkpoint (if any)."""
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done[(rec["pool"], rec["id"])] = rec["ranking"]
    return done


def merge_shards(artifact_dir: str) -> dict:
    """Merge every `progress.shard_*.jsonl` into one {(pool,id): ranking} dict."""
    merged = {}
    for p in sorted(glob.glob(os.path.join(artifact_dir, "progress.shard_*.jsonl"))):
        merged.update(load_progress(p))
    return merged


def append_progress(fh, pool: str, rid: int, ranking: list) -> None:
    fh.write(json.dumps({"pool": pool, "id": rid, "ranking": ranking}) + "\n")
    fh.flush()
    os.fsync(fh.fileno())


def assemble_submission(per_pool, out_path, top_k):
    sub = build_submission(per_pool, top_k=top_k)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(sub, f)
    return sub


def assemble_from_done(cfg, done: dict, out_path: str) -> int:
    """Build per-pool list of (rec, ranking) from a merged done-dict + cfg pools,
    write submission. Returns number of records included."""
    pools = load_pool_records(cfg.label_path, cfg.video_dir, cfg.pool_subdir)
    per_pool = {}
    n = 0
    for pool, recs in pools.items():
        items = []
        for r in recs:
            ranking = done.get((pool, r.index))
            if ranking is not None:
                items.append((r, ranking))
                n += 1
        per_pool[pool] = items
    assemble_submission(per_pool, out_path, cfg.top_k)
    return n


def run(cfg_path: str, commit_fn=None, shard_id: int = 0, n_shards: int = 1):
    """Process this shard's slice of records (i % n_shards == shard_id)."""
    cfg = load_covrx_config(cfg_path)
    commit = commit_fn or (lambda: None)
    enc = get_encoder(cfg.encoder, model=cfg.encoder_model,
                      pretrained=cfg.encoder_pretrained, num_frames=cfg.num_frames)

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

    os.makedirs(cfg.artifact_dir, exist_ok=True)
    ckpt = shard_path(cfg.artifact_dir, shard_id)
    done = load_progress(ckpt)
    print(f"[shard {shard_id}/{n_shards}] resuming: {len(done)} done in this shard")

    pools = load_pool_records(cfg.label_path, cfg.video_dir, cfg.pool_subdir)
    fh = open(ckpt, "a", encoding="utf-8")
    new_since_commit = 0
    try:
        for pool, recs in pools.items():
            pool_dir = os.path.join(cfg.video_dir, cfg.pool_subdir.get(pool, pool))
            gpath = pool_gallery_path(cfg.gallery_npz, pool)
            # gallery is shared across shards; build_gallery is idempotent
            build_gallery(list_videos(pool_dir), enc, gpath,
                          frame_loader=sample_frames, num_frames=cfg.num_frames)
            commit()
            gallery = dict(np.load(gpath))
            key_to_path = {os.path.basename(p): p for p in list_videos(pool_dir)}
            scorer = (make_qwen_scorer(proc, cfg.sample_fps, key_to_path)
                      if cfg.rerank_enabled else None)
            rk = max(1, min(cfg.rerank_k, cfg.top_k))

            for i, r in enumerate(recs):
                if i % n_shards != shard_id:
                    continue
                if (pool, r.index) in done:
                    continue
                desc = describer.describe(
                    r.video_path, r.edit_instruction).target_description
                q = enc.encode_text(desc)
                ranked = rank_pool(q, r, gallery)
                if scorer is not None:
                    head = ranked[:max(rk, cfg.top_k)]
                    mat = np.stack([gallery[k] for k in head])
                    sims = {k: float(s) for k, s in zip(head, mat @ q)}
                    # rerank only top rk; output keeps the full top_k order
                    ranked = rerank_topk(desc, ranked, sims, scorer,
                                         rk, cfg.fusion_alpha)
                append_progress(fh, pool, r.index, ranked[:cfg.top_k])
                new_since_commit += 1
                if new_since_commit >= CHECKPOINT_EVERY:
                    commit()
                    new_since_commit = 0
    finally:
        fh.close()
    commit()
    print(f"[shard {shard_id}/{n_shards}] done")


if __name__ == "__main__":
    run(sys.argv[1])
