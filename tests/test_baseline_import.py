from covrx._baseline import Record, load_records, compute_recall


def test_baseline_symbols_importable():
    assert callable(load_records)
    assert callable(compute_recall)
    rec = Record(index=0, reference_token="a", target_token="b",
                 edit_instruction="x", video_path="/tmp/a.mp4")
    assert rec.reference_key == "a.mp4"
    assert rec.target_key == "b.mp4"
