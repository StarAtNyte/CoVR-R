import numpy as np
from covrx.encoder import get_encoder, ContrastiveEncoder


def test_dummy_encoder_contract():
    enc = get_encoder("dummy", model="x", pretrained="y", num_frames=4)
    assert isinstance(enc, ContrastiveEncoder)
    v = enc.encode_video_frames([np.zeros((8, 8, 3), np.uint8)] * 4)
    t = enc.encode_text("a man rides a skateboard at night")
    assert v.shape == t.shape == (enc.dim,)
    # L2-normalized
    assert abs(np.linalg.norm(v) - 1.0) < 1e-4
    assert abs(np.linalg.norm(t) - 1.0) < 1e-4


def test_dummy_text_is_deterministic():
    enc = get_encoder("dummy", model="x", pretrained="y", num_frames=4)
    a = enc.encode_text("same string")
    b = enc.encode_text("same string")
    assert np.allclose(a, b)
