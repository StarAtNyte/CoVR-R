from dataclasses import dataclass, field, fields
from typing import Dict, Optional
import yaml

# Real HF dataset layout (orange-fox/CoVR-R): pool -> subdir under video_dir.
DEFAULT_POOL_SUBDIR = {
    "webvid": "WebVid/8M/train",
    "ss2": "something_something_v2/20bn-something-something-v2",
}


@dataclass
class CovrxConfig:
    # data
    label_path: str = ""
    video_dir: str = "./videos"
    has_labels: bool = False              # True only for self-built dev set
    pool_subdir: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_POOL_SUBDIR))

    # reasoning VLM
    vlm_model: str = "Qwen/Qwen3-VL-32B-Instruct"
    vlm_fallback_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    sample_fps: float = 1.0
    reasoning_tokens: int = 160
    description_tokens: int = 220
    n_self_consistency: int = 3
    sc_temperature: float = 0.9
    synth_temperature: float = 0.5

    # contrastive encoder
    encoder: str = "openclip"             # registry key: openclip|dummy|...
    encoder_model: str = "ViT-SO400M-16-SigLIP2-384"
    encoder_pretrained: str = "webli"     # resolve exact tag in Task 3 step 1
    num_frames: int = 8

    # retrieval / fusion / rerank
    top_k: int = 50
    fusion_alpha: float = 0.6             # final = a*retrieval + (1-a)*rerank
    dual_sim_enabled: bool = False
    dual_sim_weight: float = 0.3
    rerank_enabled: bool = True

    # io
    artifact_dir: str = "./artifacts/covrx"
    gallery_npz: str = "./artifacts/covrx/gallery.npz"
    submission_path: str = "./artifacts/covrx/submission.json"


def load_covrx_config(yaml_path: str) -> CovrxConfig:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}
    cfg = CovrxConfig()
    valid = {fld.name for fld in fields(CovrxConfig)}
    for k, v in data.items():
        if k in valid:
            setattr(cfg, k, v)
    return cfg
