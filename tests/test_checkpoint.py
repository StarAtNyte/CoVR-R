from scripts.run_submission import (
    load_progress, append_progress, shard_path, merge_shards)


def test_shard_path():
    assert shard_path("/cache/covrx/val", 3).endswith("val/progress.shard_3.jsonl")


def test_load_missing_returns_empty(tmp_path):
    assert load_progress(str(tmp_path / "nope.jsonl")) == {}


def test_append_then_load_roundtrip(tmp_path):
    p = tmp_path / "progress.jsonl"
    with open(p, "a", encoding="utf-8") as fh:
        append_progress(fh, "webvid", 0, ["a", "b"])
        append_progress(fh, "ss2", 7, ["x"])
    done = load_progress(str(p))
    assert done[("webvid", 0)] == ["a", "b"]
    assert done[("ss2", 7)] == ["x"]
    assert len(done) == 2


def test_merge_shards_combines_all(tmp_path):
    # write two shard files
    for sid, rows in [(0, [("webvid", 0, ["a"]), ("webvid", 2, ["c"])]),
                      (1, [("webvid", 1, ["b"]), ("ss2", 5, ["z"])])]:
        with open(shard_path(str(tmp_path), sid), "a", encoding="utf-8") as fh:
            for pool, rid, ranking in rows:
                append_progress(fh, pool, rid, ranking)
    merged = merge_shards(str(tmp_path))
    assert merged[("webvid", 0)] == ["a"]
    assert merged[("webvid", 1)] == ["b"]
    assert merged[("webvid", 2)] == ["c"]
    assert merged[("ss2", 5)] == ["z"]
    assert len(merged) == 4
