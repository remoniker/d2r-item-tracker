import cv2
import numpy as np
import easyocr


_reader: easyocr.Reader | None = None


def init_reader(gpu: bool = False) -> None:
    global _reader
    if gpu:
        import torch
        if not torch.cuda.is_available():
            print("Warning: --gpu requested but no CUDA device found; falling back to CPU.")
            print("  Fix: pip uninstall torch && pip install torch --index-url https://download.pytorch.org/whl/cu124")
            gpu = False
    _reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
    if gpu:
        import torch
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")


def get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def ocr_region(region: np.ndarray) -> list[str]:
    """
    OCR a D2R tooltip crop. Returns lines sorted top-to-bottom.
    Upscales 2x before passing to EasyOCR — meaningfully improves accuracy
    on the small stylized font used in D2R tooltips.
    """
    upscaled = cv2.resize(region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    results = get_reader().readtext(upscaled, detail=1, paragraph=False)
    groups = _group_into_lines(results)
    lines = []
    for group in groups:
        text = ' '.join(
            t.strip() for _, t, conf in group if t.strip() and conf > 0.2
        )
        if text:
            lines.append(text)
    return lines


def _group_into_lines(results: list) -> list[list]:
    """
    Group OCR word detections into visual lines by Y proximity, then sort
    each group left-to-right by X.

    Each group's Y anchor is fixed to the first (topmost) detection added.
    New detections join the latest group only if their top-Y is within
    tolerance of that fixed anchor — this prevents the runaway-growth bug
    where updating the anchor on every addition causes the window to drift
    downward and swallow the next visual line.
    """
    if not results:
        return []

    heights = [abs(r[0][2][1] - r[0][0][1]) for r in results]
    med_h = max(1, sorted(heights)[len(heights) // 2])
    # 40% of median glyph height: large enough to absorb sub-pixel jitter
    # on the same line, small enough that adjacent lines (spaced ~1 glyph
    # height apart) are never merged.
    tolerance = med_h * 0.4

    by_y = sorted(results, key=lambda r: r[0][0][1])

    groups: list[list] = []
    for det in by_y:
        top_y = det[0][0][1]
        # Compare against the anchor Y of the last group (first element added)
        if groups and abs(top_y - groups[-1][0][0][0][1]) <= tolerance:
            groups[-1].append(det)
        else:
            groups.append([det])

    return [sorted(g, key=lambda r: r[0][0][0]) for g in groups]
