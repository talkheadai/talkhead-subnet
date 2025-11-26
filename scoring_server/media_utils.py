import base64
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def save_video_base64_to_temp(video_b64: str, suffix: str = ".mp4") -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(base64.b64decode(video_b64))
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def probe_duration(video_path: Path) -> Optional[float]:
    """
    Use ffprobe to get video duration in seconds.
    Returns None on error.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
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
