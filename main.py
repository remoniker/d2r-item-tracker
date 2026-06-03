import argparse
import json
import os
import pathlib
import tempfile

import numpy as np

from src.extractor import extract_frames, cleanup_frames
from src.detector import detect_tooltip
from src.ocr import init_reader, ocr_region
from src.cleaner import clean_and_structure


def process_video(video_path: str, output_path: str, fps: float = 2.0, keep_frames: bool = False, gpu: bool = False):
    init_reader(gpu=gpu)
    print(f"Extracting frames at {fps} fps...")
    frame_dir = tempfile.mkdtemp(prefix="d2r_frames_")

    try:
        frames = extract_frames(video_path, frame_dir, fps=fps)
        print(f"  {len(frames)} frames extracted")

        results = []
        prev_region = None

        for i, frame_path in enumerate(frames):
            print(f"  Processing frame {i + 1}/{len(frames)}", end="\r")

            region = detect_tooltip(frame_path)

            if region is None:
                prev_region = None
                continue

            if (prev_region is not None
                    and region.shape == prev_region.shape
                    and np.array_equal(region, prev_region)):
                continue

            prev_region = region
            lines = ocr_region(region)
            if lines:
                results.append({"lines": lines})

        print()
        print(f"Found tooltips in {len(results)} frame(s)")

        # Raw OCR output
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Written to {output_path}")

        # Cleaned output
        cleaned_path = str(pathlib.Path(output_path).with_stem(pathlib.Path(output_path).stem + "_cleaned"))
        cleaned_results = [clean_and_structure(r["lines"]) for r in results]
        with open(cleaned_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_results, f, ensure_ascii=False, indent=2)
        print(f"Cleaned written to {cleaned_path}")

    finally:
        if not keep_frames:
            cleanup_frames(frame_dir)


def main():
    parser = argparse.ArgumentParser(description="Extract D2R tooltip text from a video clip")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("-o", "--output", default="items.json", help="Output JSON file (default: items.json)")
    parser.add_argument("--fps", type=float, default=2.0, help="Frame sampling rate (default: 2.0)")
    parser.add_argument("--keep-frames", action="store_true", help="Keep extracted frames for debugging")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for EasyOCR (much faster if available)")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"Error: file not found: {args.video}")
        return

    process_video(args.video, args.output, fps=args.fps, keep_frames=args.keep_frames, gpu=args.gpu)


if __name__ == "__main__":
    main()
