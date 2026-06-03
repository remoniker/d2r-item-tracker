import cv2
import numpy as np


_BLACK_THRESH = 10       # pixels darker than this are tooltip background
_TEXT_THRESH = 80        # pixels brighter than this are text
_MIN_RECT_FILL = 0.60    # dark pixel fraction of bounding rect
_MIN_W = 100
_MIN_H = 60
_MAX_W = 900
_MAX_H = 950
_MIN_TEXT_FRACTION = 0.01
_MAX_TEXT_FRACTION = 0.60


def detect_tooltip(frame_path: str) -> np.ndarray | None:
    """Return a cropped BGR image of the tooltip, or None if not found."""
    img = cv2.imread(frame_path)
    if img is None:
        return None
    bounds = _find_tooltip_rect(img)
    if bounds is None:
        return None
    x, y, w, h = bounds
    return img[y:y + h, x:x + w]


def _find_tooltip_rect(img: np.ndarray) -> tuple | None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # No morphological close — a large kernel merges the tooltip with the dark
    # game-world background, producing a giant contour that fails every filter.
    # The raw near-black mask is sufficient because tooltip backgrounds are
    # dense pure-black with only sparse bright text pixels breaking them up.
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
    return best[:4]
