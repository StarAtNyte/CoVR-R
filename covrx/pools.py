import json
import os
from dataclasses import dataclass
from typing import Dict, List

POOL_EXT = {"webvid": ".mp4", "ss2": ".webm"}


@dataclass
class PoolRecord:
    index: int
    pool: str
    reference_token: str
    target_token: str
    edit_instruction: str
    video_path: str
    ext: str

    @property
    def reference_key(self) -> str:
        return f"{self.reference_token}{self.ext}"

    @property
    def target_key(self) -> str:
        return f"{self.target_token}{self.ext}"

    @property
    def has_label(self) -> bool:
        return bool(self.target_token)


def load_pool_records(label_path: str, video_dir: str) -> Dict[str, List[PoolRecord]]:
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]

    pools: Dict[str, List[PoolRecord]] = {}
    for section in data:
        for pool_key, entries in section.items():
            ext = POOL_EXT.get(pool_key, ".mp4")
            bucket = pools.setdefault(pool_key, [])
            for i, row in enumerate(entries):
                src = str(row.get("video_source", "")).strip()
                tgt = str(row.get("video_target", "")).strip()
                ref_token = os.path.basename(src).split(".")[0]
                tgt_token = os.path.basename(tgt).split(".")[0] if tgt else ""
                src_rel = os.path.splitext(src)[0]  # "112/1016223889" (strip any ext)
                bucket.append(PoolRecord(
                    index=int(row.get("id", row.get("index", i))),
                    pool=pool_key,
                    reference_token=ref_token,
                    target_token=tgt_token,
                    edit_instruction=(row.get("modification_text") or "").strip(),
                    video_path=os.path.join(video_dir, pool_key, f"{src_rel}{ext}"),
                    ext=ext,
                ))
    return pools
