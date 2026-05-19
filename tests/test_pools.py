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


def test_video_path_preserves_source_subdir(tmp_path):
    import json, os
    from covrx.pools import load_pool_records
    p = tmp_path / "v.json"
    p.write_text(json.dumps([{"webvid": [
        {"id": 0, "video_source": "112/1016223889",
         "modification_text": "x"}]}]))
    r = load_pool_records(p.as_posix(), "/data/videos")["webvid"][0]
    assert r.video_path == os.path.join("/data/videos", "webvid", "112", "1016223889.mp4")
    assert r.reference_key == "1016223889.mp4"   # gallery key still basename


def test_pool_subdir_mapping_real_hf_layout(tmp_path):
    import json, os
    from covrx.pools import load_pool_records
    from covrx.config import DEFAULT_POOL_SUBDIR
    p = tmp_path / "v.json"
    p.write_text(json.dumps([
        {"webvid": [{"id": 0, "video_source": "112/1016223889",
                     "modification_text": "x"}]},
        {"ss2": [{"id": 1, "video_source": 74225,
                  "modification_text": "y"}]},
    ]))
    pools = load_pool_records(p.as_posix(), "/cache/covrr", DEFAULT_POOL_SUBDIR)
    wv = pools["webvid"][0]
    ss = pools["ss2"][0]
    assert wv.video_path == os.path.join(
        "/cache/covrr", "WebVid/8M/train", "112", "1016223889.mp4")
    assert wv.reference_key == "1016223889.mp4"        # gallery key = basename
    assert ss.video_path == os.path.join(
        "/cache/covrr", "something_something_v2/20bn-something-something-v2",
        "74225.webm")
    assert ss.reference_key == "74225.webm"
