import hashlib
import json


def _item_hash(item: dict) -> str:
    key = item.get("name", "") + "|" + "|".join(l["text"] for l in item.get("lines", []))
    return hashlib.md5(key.encode()).hexdigest()


def deduplicate(items: list[dict]) -> list[dict]:
    """
    Remove items that are identical across frames.
    Preserves order and keeps the first occurrence.
    """
    seen = set()
    unique = []
    for item in items:
        h = _item_hash(item)
        if h not in seen:
            seen.add(h)
            unique.append(item)
    return unique
