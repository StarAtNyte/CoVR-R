import json
from scripts.run_submission import assemble_submission
from covrx.pools import PoolRecord


def test_assemble_submission_writes_expected_keys(tmp_path):
    rec = PoolRecord(7, "webvid", "ref", "", "edit",
                     "./v/webvid/ref.mp4", ".mp4")
    per_pool = {"webvid": [(rec, ["t1.mp4", "t2.mp4"])]}
    out = tmp_path / "sub.json"
    assemble_submission(per_pool, str(out), top_k=50)
    data = json.loads(out.read_text())
    assert data["webvid"][0]["id"] == 7
    assert data["webvid"][0]["ranking"][:2] == ["t1", "t2"]
