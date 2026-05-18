import numpy as np
from covrx.index import build_gallery
from covrx.encoder import get_encoder


def test_build_and_resume(tmp_path):
    enc = get_encoder("dummy")
    frames = {"a.mp4": [np.zeros((4, 4, 3), np.uint8)],
              "b.mp4": [np.ones((4, 4, 3), np.uint8)]}

    def fake_loader(path, n):
        return frames[path.split("/")[-1]]

    npz = tmp_path / "g.npz"
    n1 = build_gallery(["x/a.mp4", "x/b.mp4"], enc, str(npz),
                        frame_loader=fake_loader, num_frames=1)
    assert n1 == 2
    store = dict(np.load(str(npz)))
    assert set(store) == {"a.mp4", "b.mp4"}
    assert abs(np.linalg.norm(store["a.mp4"]) - 1.0) < 1e-4

    # resume: only c.mp4 is new
    frames["c.mp4"] = [np.full((4, 4, 3), 7, np.uint8)]
    n2 = build_gallery(["x/a.mp4", "x/b.mp4", "x/c.mp4"], enc, str(npz),
                        frame_loader=fake_loader, num_frames=1)
    assert n2 == 1   # only c encoded
    store = dict(np.load(str(npz)))
    assert set(store) == {"a.mp4", "b.mp4", "c.mp4"}
