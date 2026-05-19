import numpy as np
import covrx.frames as F
from covrx.frames import uniform_indices, sample_frames_from_array


def test_uniform_indices_basic():
    assert uniform_indices(100, 5) == [0, 24, 49, 74, 99]
    assert uniform_indices(3, 5) == [0, 1, 2, 2, 2]   # clamp when short
    assert uniform_indices(0, 5) == [0, 0, 0, 0, 0]


def test_sample_from_array():
    vid = np.arange(10 * 4 * 4 * 3, dtype=np.uint8).reshape(10, 4, 4, 3)
    out = sample_frames_from_array(vid, 4)
    assert len(out) == 4
    assert out[0].shape == (4, 4, 3)
    assert np.array_equal(out[0], vid[0])
    assert np.array_equal(out[-1], vid[9])


def test_sample_frames_falls_back_to_pyav(monkeypatch):
    sentinel = [np.zeros((2, 2, 3), np.uint8)]

    def boom(path, n):
        raise RuntimeError("decord cannot decode vp9")

    monkeypatch.setattr(F, "_sample_decord", boom)
    monkeypatch.setattr(F, "_sample_pyav", lambda path, n: sentinel)
    assert F.sample_frames("whatever.webm", 4) is sentinel


def test_sample_frames_prefers_decord(monkeypatch):
    monkeypatch.setattr(F, "_sample_decord", lambda p, n: ["decord"])
    monkeypatch.setattr(F, "_sample_pyav",
                        lambda p, n: (_ for _ in ()).throw(AssertionError))
    assert F.sample_frames("x.mp4", 4) == ["decord"]
