import os
from typing import Callable, Dict, List
import numpy as np

FrameLoader = Callable[[str, int], list]


def _load_existing(npz_path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(npz_path):
        return {}
    with np.load(npz_path) as z:
        return {k: np.asarray(z[k], np.float32) for k in z.files}


def build_gallery(video_paths: List[str], encoder, npz_path: str,
                  frame_loader: FrameLoader, num_frames: int = 8) -> int:
    os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
    store = _load_existing(npz_path)
    encoded = 0
    for vp in video_paths:
        key = os.path.basename(vp)
        if key in store:
            continue
        try:
            frames = frame_loader(vp, num_frames)
            store[key] = encoder.encode_video_frames(frames).astype(np.float32)
            encoded += 1
        except Exception as e:  # missing/corrupt video → skip + log
            print(f"[index] skip {vp}: {e}")
        if encoded and encoded % 200 == 0:
            np.savez_compressed(npz_path, **store)
    np.savez_compressed(npz_path, **store)
    return encoded


def pool_gallery_path(base_npz: str, pool: str) -> str:
    root, ext = os.path.splitext(base_npz)
    return f"{root}.{pool}{ext or '.npz'}"
