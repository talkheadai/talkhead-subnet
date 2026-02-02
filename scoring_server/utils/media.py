import base64
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import cv2
import numpy as np
import requests


def download_video_from_url(video_url: str) -> Path:
    """
    Download a remote video to a temporary file and return its path.
    """
    resp = requests.get(video_url, stream=True, timeout=30)
    resp.raise_for_status()

    parsed = urlparse(video_url)
    suffix = Path(parsed.path).suffix or ".mp4"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                tmp.write(chunk)
        tmp_path = Path(tmp.name)
    return tmp_path


def probe_duration(video_path: Path) -> Optional[float]:
    """
    Use ffprobe to get video duration in seconds.
    Returns None on error.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-hide_banner",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return float(out.decode().strip())
    except Exception:
        return None


def extract_audio(video_path: Path) -> Optional[Path]:
    """
    Extract audio to a temporary WAV file using ffmpeg.
    Returns Path or None on error.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-nostats",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(tmp_path),
    ]
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return tmp_path
    except Exception:
        return None


def _sample_frames(video_path: Path, max_frames: int = 32) -> list[np.ndarray]:
    """
    Sample up to max_frames frames evenly from the video.
    Returns list of BGR images (as numpy arrays).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        cap.release()
        return []

    step = max(1, frame_count // max_frames)
    frames = []
    idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if len(frames) >= max_frames:
            break
        idx += step

    cap.release()
    return frames