import json
from covrx.devset import to_pool_json


def test_to_pool_json_shape(tmp_path):
    rows = [{"id": 1, "video_source": "1/a", "video_target": "1/b",
             "modification_text": "edit one"}]
    out = to_pool_json(rows, pool="webvid")
    assert out == [{"webvid": rows}]
    p = tmp_path / "dev.json"
    p.write_text(json.dumps(out))
    from covrx.pools import load_pool_records
    rec = load_pool_records(str(p), "./videos")["webvid"][0]
    assert rec.has_label and rec.target_token == "b"
