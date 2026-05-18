"""Shim to import the authors' baseline modules without modifying them."""
import os
import sys

BASELINE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "CoVR-R-authors-github",
)
if BASELINE_DIR not in sys.path:
    sys.path.insert(0, BASELINE_DIR)

import data as baseline_data          # noqa: E402
import evaluate as baseline_evaluate  # noqa: E402

Record = baseline_data.Record
load_records = baseline_data.load_records
list_videos = baseline_data.list_videos
compute_recall = baseline_evaluate.compute_recall
