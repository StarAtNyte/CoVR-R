from scripts.run_submission import (
    load_progress, append_progress, progress_path)


def test_progress_path():
    assert progress_path("/cache/covrx/val").endswith("val/progress.jsonl")


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


def test_resume_skips_done(tmp_path):
    p = tmp_path / "progress.jsonl"
    with open(p, "a", encoding="utf-8") as fh:
        append_progress(fh, "webvid", 3, ["t1", "t2"])
    done = load_progress(str(p))
    # simulate the run() skip check
    assert done.get(("webvid", 3)) == ["t1", "t2"]
    assert done.get(("webvid", 99)) is None
