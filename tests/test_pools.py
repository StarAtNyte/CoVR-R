import json
from covrx.pools import load_pool_records, PoolRecord


def _write(p, obj):
    p.write_text(json.dumps(obj))
    return str(p)


def test_no_labels_merged(tmp_path):
    path = _write(tmp_path / "val.json", [
        {"webvid": [{"id": 0, "video_source": "112/1016223889",
                     "modification_text": "make it night"}]},
        {"ss2": [{"id": 1, "video_source": "777/abc",
                  "modification_text": "add a dog"}]},
    ])
    pools = load_pool_records(path, "./videos")
    assert set(pools) == {"webvid", "ss2"}
    wv = pools["webvid"][0]
    assert isinstance(wv, PoolRecord)
    assert wv.reference_token == "1016223889"
    assert wv.reference_key == "1016223889.mp4"
    assert wv.edit_instruction == "make it night"
    assert wv.has_label is False
    assert wv.target_token == ""
    ss = pools["ss2"][0]
    assert ss.reference_key == "abc.webm"


def test_with_labels_devset(tmp_path):
    path = _write(tmp_path / "dev.json", [
        {"webvid": [{"id": 5, "video_source": "1/x", "video_target": "1/y",
                     "modification_text": "edit"}]},
    ])
    pools = load_pool_records(path, "./videos")
    r = pools["webvid"][0]
    assert r.has_label is True
    assert r.target_token == "y"
    assert r.target_key == "y.mp4"
