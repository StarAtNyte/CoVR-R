from typing import List, Dict


def to_pool_json(rows: List[dict], pool: str) -> list:
    """Wrap labeled external triplets into the merged pool JSON shape."""
    return [{pool: rows}]
