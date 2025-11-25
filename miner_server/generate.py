from typing import  Optional
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Put a small sample video here later: data/sample_video.mp4
SAMPLE_VIDEO_PATH = ROOT / "data" / "sample_video.mp4"


def generate_talking_head(
    image_bytes: bytes,
    script: str,
    duration_sec: Optional[float] = None,
) -> bytes:
    """
    Core generation stub.

    Later: run TTS + talking-head model here.

    For now: return the bytes of a placeholder MP4 file so
    the API and validator plumbing can be built & tested.
    """
    if not SAMPLE_VIDEO_PATH.exists():
        # For now, just clearly fail if you forgot to add sample video.
        raise FileNotFoundError(
            f"Placeholder video not found at {SAMPLE_VIDEO_PATH}. "
            "Drop a small .mp4 there or implement real generation."
        )

    return SAMPLE_VIDEO_PATH.read_bytes()
