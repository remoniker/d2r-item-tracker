import os
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

FONT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "films.EXH_____.ttf",
)

# Only characters that actually appear in D2R tooltip text.
# Deliberately excludes []{}|~@#^_`<>=?\ which cause false positives
# by matching vertical strokes and serifs inside real letters.
CHARS = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    "+'\"*-%./:,()"
)

_MATCH_THRESHOLD = 0.65
_MIN_PATCH_BRIGHTNESS = 25   # skip positions where image patch is nearly dark
_SPACE_FACTOR = 0.40         # gap > font_size * factor → insert space

# Cap height as a fraction of font point size for this font (measured empirically)
_CAP_HEIGHT_RATIO = 0.60


def _render_glyph(font: ImageFont.FreeTypeFont, ch: str) -> np.ndarray | None:
    try:
        bbox = font.getbbox(ch)
    except Exception:
        return None
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if w <= 0 or h <= 0:
        return None
    pad = 1
    canvas = Image.new("L", (w + pad * 2, h + pad * 2), 0)
    ImageDraw.Draw(canvas).text((pad - bbox[0], pad - bbox[1]), ch, font=font, fill=255)
    arr = np.array(canvas)
    return arr if arr.max() > 0 else None


def build_templates(font_size: int) -> dict[str, np.ndarray]:
    font = ImageFont.truetype(FONT_PATH, font_size)
    return {ch: g for ch in CHARS if (g := _render_glyph(font, ch)) is not None}


def _estimate_font_size(gray: np.ndarray) -> int:
    _, bright = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    row_has_text = (bright.sum(axis=1) / bright.shape[1]) > 0.005
    heights = []
    start = None
    for i, v in enumerate(row_has_text):
        if v and start is None:
            start = i
        elif not v and start is not None:
            heights.append(i - start)
            start = None
    if start is not None:
        heights.append(len(row_has_text) - start)
    if not heights:
        return 30
    median_h = sorted(heights)[len(heights) // 2]
    return max(12, min(60, round(median_h / _CAP_HEIGHT_RATIO)))


def _find_text_lines(gray: np.ndarray) -> list[tuple[int, int]]:
    _, bright = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    row_has_text = (bright.sum(axis=1) / bright.shape[1]) > 0.005
    bands: list[list[int]] = []
    start = None
    for i, v in enumerate(row_has_text):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= 4:
                bands.append([start, i])
            start = None
    if start is not None and gray.shape[0] - start >= 4:
        bands.append([start, gray.shape[0]])
    merged: list[list[int]] = []
    for s, e in bands:
        if merged and s - merged[-1][1] <= 3:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _match_line(line_gray: np.ndarray, templates: dict[str, np.ndarray], font_size: int) -> str:
    if line_gray.max() == 0:
        return ""

    # Pad line to at least font_size height so templates always fit vertically
    h = line_gray.shape[0]
    if h < font_size:
        pad_top = (font_size - h) // 2
        pad_bot = font_size - h - pad_top
        line_gray = np.pad(line_gray, ((pad_top, pad_bot), (0, 0)))
    target_h = line_gray.shape[0]
    img_f = line_gray.astype(np.float32)

    # Precompute column max brightness for fast patch filtering
    col_max = img_f.max(axis=0)

    all_matches: list[tuple[int, int, str, float]] = []

    for ch, tmpl in templates.items():
        th, tw = tmpl.shape
        sw = max(1, round(tw * target_h / th))
        if sw > img_f.shape[1]:
            continue
        scaled = cv2.resize(tmpl, (sw, target_h), interpolation=cv2.INTER_LINEAR)
        result = cv2.matchTemplate(img_f, scaled.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        scores = result[0]
        for x in np.where(scores >= _MATCH_THRESHOLD)[0]:
            # Reject matches in low-brightness regions (dark background false positives)
            if col_max[x : x + sw].max() < _MIN_PATCH_BRIGHTNESS:
                continue
            all_matches.append((int(x), int(x) + sw, ch, float(scores[x])))

    if not all_matches:
        return ""

    # NMS: sort by score desc, mark used columns
    all_matches.sort(key=lambda m: -m[3])
    kept: list[tuple[int, int, str]] = []
    used = np.zeros(img_f.shape[1], dtype=bool)

    for x_l, x_r, ch, _ in all_matches:
        mid = (x_l + x_r) // 2
        if not used[mid]:
            kept.append((x_l, x_r, ch))
            used[x_l:x_r] = True

    kept.sort(key=lambda m: m[0])

    space_gap = font_size * _SPACE_FACTOR
    chars: list[str] = []
    prev_right: int | None = None

    for x_l, x_r, ch in kept:
        if prev_right is not None and x_l - prev_right > space_gap:
            chars.append(" ")
        chars.append(ch)
        prev_right = x_r

    return "".join(chars)


def ocr_region(region: np.ndarray) -> list[str]:
    """
    OCR a D2R tooltip crop using font template matching.
    Returns one string per text line, top to bottom.
    """
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region.copy()
    font_size = _estimate_font_size(gray)
    templates = build_templates(font_size)
    lines = []
    for y0, y1 in _find_text_lines(gray):
        text = _match_line(gray[y0:y1, :], templates, font_size)
        if text.strip():
            lines.append(text.strip())
    return lines
