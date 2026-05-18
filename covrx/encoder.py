# Resolved open_clip tag (Task 3 step 1): ('ViT-SO400M-16-SigLIP2-384', 'webli')
import hashlib
from abc import ABC, abstractmethod
from typing import List
import numpy as np

_REGISTRY = {}


def register(name):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


class ContrastiveEncoder(ABC):
    dim: int = 0

    @abstractmethod
    def encode_video_frames(self, frames: List[np.ndarray]) -> np.ndarray: ...

    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray: ...


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x) + 1e-8
    return (x / n).astype(np.float32)


@register("dummy")
class DummyEncoder(ContrastiveEncoder):
    """Deterministic hash-based encoder for fast CPU tests. No model."""
    dim = 32

    def __init__(self, **_):
        pass

    def _hash_vec(self, key: str) -> np.ndarray:
        h = hashlib.sha256(key.encode()).digest()
        arr = np.frombuffer((h * 2)[: self.dim], dtype=np.uint8).astype(np.float32)
        return _l2(arr - 127.5)

    def encode_video_frames(self, frames):
        return self._hash_vec(f"v{len(frames)}-{int(frames[0].sum())}")

    def encode_text(self, text):
        return self._hash_vec(f"t{text}")


@register("openclip")
class OpenClipEncoder(ContrastiveEncoder):
    """SigLIP2 frame-mean video embedding + text embedding (shared space)."""

    def __init__(self, model: str, pretrained: str, num_frames: int = 8, device: str = "cuda"):
        import open_clip
        import torch
        self.torch = torch
        self.num_frames = num_frames
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model, pretrained=pretrained
        )
        self.model = self.model.eval().to(device)
        self.tokenizer = open_clip.get_tokenizer(model)
        with torch.no_grad():
            d = self.model.encode_text(self.tokenizer(["x"]).to(device)).shape[-1]
        self.dim = int(d)

    def encode_video_frames(self, frames):
        from PIL import Image
        torch = self.torch
        imgs = [self.preprocess(Image.fromarray(f)).unsqueeze(0) for f in frames]
        batch = torch.cat(imgs).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            v = feats.mean(0)
        return _l2(v.float().cpu().numpy())

    def encode_text(self, text):
        torch = self.torch
        toks = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            t = self.model.encode_text(toks)[0]
        return _l2(t.float().cpu().numpy())


def get_encoder(name: str, **kwargs) -> ContrastiveEncoder:
    if name not in _REGISTRY:
        raise KeyError(f"unknown encoder {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
