from typing import List
import numpy as np


def uniform_indices(n_total: int, n_want: int) -> List[int]:
    if n_want <= 0:
        return []
    if n_total <= 0 or n_want == 1:
        return [0] * n_want
    if n_total < n_want:                      # short clip: list then pad w/ last
        idx = list(range(n_total))
        return idx + [n_total - 1] * (n_want - n_total)
    return [int(i * (n_total - 1) / (n_want - 1)) for i in range(n_want)]


def sample_frames_from_array(video: np.ndarray, n_want: int) -> List[np.ndarray]:
    idx = uniform_indices(len(video), n_want)
    return [video[i] for i in idx]


def sample_frames(video_path: str, n_want: int) -> List[np.ndarray]:
    """Decode a video file and return n_want uniformly-spaced RGB frames."""
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path)
    idx = uniform_indices(len(vr), n_want)
    return [vr[i].asnumpy() for i in idx]
