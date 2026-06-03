def parse_item(lines: list[dict]) -> dict:
    if not lines:
        return {}

    name_line = lines[0]
    stat_lines = lines[1:]

    return {
        "name": name_line["text"],
        "lines": [
            {"text": l["text"], "confidence": l["confidence"]}
            for l in stat_lines
        ],
    }
