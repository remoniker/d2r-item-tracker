import pathlib

_DATA_FILE = pathlib.Path(__file__).parent.parent / "item_names.txt"


def _load() -> frozenset[str]:
    if not _DATA_FILE.exists():
        return frozenset()
    with open(_DATA_FILE, encoding="utf-8") as f:
        return frozenset(line.strip().upper() for line in f if line.strip())


ITEM_NAMES: frozenset[str] = _load()
