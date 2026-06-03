import json
import os
import cv2
import numpy as np

from src.font_ocr import ocr_region

FRAMES_DIR = "frames_with_largefont"
OUTPUT_JSON = "tooltips.json"

# Tooltip detection parameters
_BLACK_THRESH = 10
_TEXT_THRESH = 80
_MIN_RECT_FILL = 0.60
_MIN_W = 100
_MIN_H = 60
_MAX_W = 900
_MAX_H = 950
_MIN_TEXT_FRACTION = 0.01
_MAX_TEXT_FRACTION = 0.60


def find_tooltip(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # No morphological close — the large close kernel merges the tooltip with
    # the dark game background, producing one giant contour that gets discarded.
    # The raw black mask is sufficient because tooltip backgrounds are dense-black.
    _, black = cv2.threshold(gray, _BLACK_THRESH, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < _MIN_W or h < _MIN_H or w > _MAX_W or h > _MAX_H:
            continue
        interior = gray[y:y + h, x:x + w]
        dark_fraction = np.count_nonzero(interior < _BLACK_THRESH) / interior.size
        if dark_fraction < _MIN_RECT_FILL:
            continue
        _, bright = cv2.threshold(interior, _TEXT_THRESH, 255, cv2.THRESH_BINARY)
        text_fraction = np.count_nonzero(bright) / bright.size
        if text_fraction < _MIN_TEXT_FRACTION or text_fraction > _MAX_TEXT_FRACTION:
            continue
        candidates.append((x, y, w, h, w * h))

    if not candidates:
        return None

    best = max(candidates, key=lambda c: c[4])
    x, y, w, h = best[:4]
    return img[y:y + h, x:x + w]


def main():
    frame_files = sorted(
        f for f in os.listdir(FRAMES_DIR) if f.endswith(".png")
    )

    results = []
    for fname in frame_files:
        path = os.path.join(FRAMES_DIR, fname)
        img = cv2.imread(path)
        if img is None:
            continue

        region = find_tooltip(img)
        if region is None:
            print(f"{fname}: no tooltip")
            continue

        text_lines = ocr_region(region)
        if not text_lines:
            print(f"{fname}: tooltip found but no text")
            continue

        line = " | ".join(text_lines)
        results.append({"frame": fname, "tooltip": line})
        print(f"{fname}: {line}")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(results)} tooltip(s) to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
