import numpy as np
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
