import os
import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install_from_requirements(os.path.join(ROOT, "requirements-covrx.txt"))
    .add_local_dir(os.path.join(ROOT, "covrx"), "/root/covrx")
    .add_local_dir(os.path.join(ROOT, "CoVR-R-authors-github"), "/root/CoVR-R-authors-github")
)

app = modal.App("covrx")
cache = modal.Volume.from_name("covrx-cache", create_if_missing=True)
CACHE = "/cache"


@app.cls(gpu="A100-80GB", image=image, volumes={CACHE: cache},
         timeout=60 * 60 * 6, scaledown_window=300)
class Pipeline:
    cfg_yaml: str = modal.parameter()

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
    def index_pool(self, video_paths: list[str]) -> int:
        from covrx.index import build_gallery
        from covrx.frames import sample_frames
        return build_gallery(video_paths, self.encoder, self.cfg.gallery_npz,
                              frame_loader=sample_frames,
                              num_frames=self.cfg.num_frames)

    @modal.method()
    def reason_describe(self, items: list[dict]) -> list[dict]:
        """items: [{'video': path, 'edit': str, 'id': int, 'pool': str}]"""
        import sys
        sys.path.insert(0, "/root/CoVR-R-authors-github")
        from config import Config
        from model import VideoProcessor
        from covrx.reason import ReasonedDescriber, make_qwen_generate
        bcfg = Config(model_name=self.cfg.vlm_model, parallelism="mp")
        proc = VideoProcessor(bcfg)
        proc.load_model(0)
        rd = ReasonedDescriber(make_qwen_generate(proc, self.cfg.sample_fps),
                               self.cfg.n_self_consistency)
        out = []
        for it in items:
            r = rd.describe(it["video"], it["edit"])
            out.append({**it, "target_description": r.target_description,
                        "reasoning": r.reasoning})
        return out


@app.local_entrypoint()
def main(stage: str, cfg: str):
    p = Pipeline(cfg_yaml=cfg)
    if stage == "index":
        import sys
        sys.path.insert(0, os.path.join(ROOT, "covrx"))
        from covrx.config import load_covrx_config
        from covrx._baseline import list_videos
        c = load_covrx_config(cfg)
        print("indexed:", p.index_pool.remote(list_videos(c.video_dir)))
