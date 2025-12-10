import subprocess
import tempfile
from pathlib import Path
from typing import Optional


# Adjust this to wherever you cloned SadTalker
SADTALKER_ROOT = Path(__file__).resolve().parents[1] / "miner_server" / "SadTalker"
INFERENCE_SCRIPT = SADTALKER_ROOT / "inference.py"


def generate_video_with_sadtalker(
    image_bytes: bytes,
    audio_path: Path,
) -> bytes:
    """
    Wrap SadTalker's CLI:
      python inference.py --driven_audio <audio> --source_image <img> --result_dir <out>

    Returns the MP4 bytes of the generated talking-head video.
    """
    if not INFERENCE_SCRIPT.exists():
        raise FileNotFoundError(
            f"SadTalker inference script not found at {INFERENCE_SCRIPT}. "
            "Check SADTALKER_ROOT path."
        )

    # 1. Create a temporary working directory
    work_dir = Path(tempfile.mkdtemp(prefix="talkhead_sadtalker_"))

    # 2. Save image bytes into work_dir
    source_image_path = work_dir / "source.png"
    source_image_path.write_bytes(image_bytes)

    # 3. Choose result directory
    result_dir = work_dir / "results"
    result_dir.mkdir(exist_ok=True, parents=True)

    # 4. Build SadTalker CLI command
    cmd = [
        "python",
        str(INFERENCE_SCRIPT),
        "--driven_audio", str(audio_path),
        "--source_image", str(source_image_path),
        "--result_dir", str(result_dir),
        "--preprocess", "crop",     # common default
        "--still",                  # good for single portrait
        "--enhancer", "gfpgan",     # if you have enhancer weights; otherwise remove
    ]

    # SadTalker generally decides duration from audio; duration_sec is optional.
    # If you really want to clamp, you can trim audio before calling SadTalker.

    # 5. Run SadTalker
    proc = subprocess.run(
        cmd,
        cwd=str(SADTALKER_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"SadTalker failed with code {proc.returncode}:\n"
            f"{proc.stderr.decode('utf-8', errors='ignore')}"
        )

    # 6. Find the output video (SadTalker usually creates an mp4 in result_dir)
    mp4_candidates = list(result_dir.glob("*.mp4"))
    if not mp4_candidates:
        raise FileNotFoundError(
            f"No MP4 output found in {result_dir} after SadTalker run."
        )

    # If multiple, pick the latest
    mp4_path = sorted(mp4_candidates, key=lambda p: p.stat().st_mtime)[-1]
    return mp4_path.read_bytes()
