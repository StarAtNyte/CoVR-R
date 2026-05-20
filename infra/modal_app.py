import os
import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install_from_requirements(os.path.join(ROOT, "requirements-covrx.txt"))
    .add_local_dir(os.path.join(ROOT, "covrx"), "/root/covrx")
    .add_local_dir(os.path.join(ROOT, "scripts"), "/root/scripts")
    .add_local_dir(os.path.join(ROOT, "configs"), "/root/configs")
    .add_local_dir(os.path.join(ROOT, "CoVR-R-authors-github"), "/root/CoVR-R-authors-github")
)

app = modal.App("covrx")
cache = modal.Volume.from_name("covrx-cache", create_if_missing=True)
CACHE = "/cache"
DATA = "/cache/covrr"


@app.function(image=image, volumes={CACHE: cache}, timeout=60 * 60 * 6)
def ingest(repo_id: str = "orange-fox/CoVR-R") -> list[str]:
    """Download the HF dataset straight into the persistent Volume (no GPU)."""
    from huggingface_hub import snapshot_download
    os.environ["HF_HOME"] = f"{CACHE}/hf"
    snapshot_download(repo_id, repo_type="dataset", local_dir=DATA,
                      max_workers=16)
    cache.commit()
    listing = []
    for root, _, files in os.walk(DATA):
        for f in sorted(files)[:5]:
            listing.append(os.path.relpath(os.path.join(root, f), DATA))
    return listing[:50]


# Pipeline parameters: cfg path + (shard_id, n_shards) so one container = one shard.
@app.cls(gpu="A100-80GB", image=image, volumes={CACHE: cache},
         timeout=60 * 60 * 24, scaledown_window=300)
class Pipeline:
    cfg_yaml: str = modal.parameter()
    shard_id: int = modal.parameter(default=0)
    n_shards: int = modal.parameter(default=1)

    @modal.enter()
    def setup(self):
        os.environ["HF_HOME"] = f"{CACHE}/hf"
        import sys
        sys.path.insert(0, "/root")
        from covrx.config import load_covrx_config
        from covrx.encoder import get_encoder
        self.cfg = load_covrx_config(self.cfg_yaml)
        self.encoder = get_encoder(
            self.cfg.encoder, model=self.cfg.encoder_model,
            pretrained=self.cfg.encoder_pretrained,
            num_frames=self.cfg.num_frames)

    @modal.method()
    def warm_index(self) -> dict:
        import sys
        sys.path.insert(0, "/root")
        from covrx.index import build_gallery, pool_gallery_path
        from covrx.frames import sample_frames
        from covrx._baseline import list_videos
        counts = {}
        for pool in ("webvid", "ss2"):
            pool_dir = os.path.join(self.cfg.video_dir,
                                    self.cfg.pool_subdir.get(pool, pool))
            if not os.path.isdir(pool_dir):
                counts[pool] = -1
                continue
            n = build_gallery(
                list_videos(pool_dir), self.encoder,
                pool_gallery_path(self.cfg.gallery_npz, pool),
                frame_loader=sample_frames, num_frames=self.cfg.num_frames)
            counts[pool] = n
        cache.commit()
        return counts

    @modal.method()
    def process_shard(self) -> str:
        import sys
        sys.path.insert(0, "/root")
        sys.path.insert(0, "/root/CoVR-R-authors-github")
        from scripts.run_submission import run
        run(self.cfg_yaml, commit_fn=cache.commit,
            shard_id=self.shard_id, n_shards=self.n_shards)
        cache.commit()
        return f"shard {self.shard_id}/{self.n_shards} done"


@app.function(image=image, volumes={CACHE: cache}, timeout=60 * 60 * 2)
def assemble(cfg_yaml: str) -> dict:
    """Merge all shard checkpoints on the Volume into the final submission JSON."""
    import sys
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/CoVR-R-authors-github")
    from covrx.config import load_covrx_config
    from scripts.run_submission import merge_shards, assemble_from_done
    cfg = load_covrx_config(cfg_yaml)
    done = merge_shards(cfg.artifact_dir)
    n = assemble_from_done(cfg, done, cfg.submission_path)
    cache.commit()
    return {"records": n, "submission": cfg.submission_path}


@app.local_entrypoint()
def main(stage: str, cfg: str = "configs/val.yaml", shards: int = 10):
    if stage == "download":
        print("dataset tree sample:")
        for p in ingest.remote():
            print(" ", p)
        return
    if stage == "index":
        pipe = Pipeline(cfg_yaml=cfg)
        print("indexed:", pipe.warm_index.remote())
        return
    if stage == "psubmit":
        # Fan out `shards` parallel containers, each processes its slice.
        pipes = [Pipeline(cfg_yaml=cfg, shard_id=i, n_shards=shards)
                 for i in range(shards)]
        # .map across instances by spawning each and waiting
        futures = [p.process_shard.spawn() for p in pipes]
        for f in futures:
            print(f.get())
        print("merging shards into submission ...")
        print(assemble.remote(cfg))
        return
    if stage == "assemble":
        print(assemble.remote(cfg))
        return
    if stage == "submit":          # single-container path (legacy / smoke)
        pipe = Pipeline(cfg_yaml=cfg)
        print("submission written to Volume:", pipe.process_shard.remote())
        return
    raise SystemExit(f"unknown stage {stage!r}; use download|index|psubmit|assemble|submit")
