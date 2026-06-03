import subprocess
import os
import shutil
from pathlib import Path


def check_ffmpeg():
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found in PATH. Install from https://ffmpeg.org/download.html")


def extract_frames(video_path: str, output_dir: str, fps: float = 2.0) -> list[str]:
    """
    Extract frames from video at given fps using ffmpeg.
    Returns sorted list of frame file paths.
    """
    check_ffmpeg()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pattern = os.path.join(output_dir, "frame_%05d.png")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        pattern,
        "-y",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    frames = sorted(Path(output_dir).glob("frame_*.png"))
    return [str(f) for f in frames]


def cleanup_frames(output_dir: str):
    shutil.rmtree(output_dir, ignore_errors=True)
