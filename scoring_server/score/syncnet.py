from functools import lru_cache
from pathlib import Path
from typing import Tuple

import torch
from syncnet_python import SyncNetPipeline


WEIGHTS_DIR = Path(__file__).resolve().parents[0] / "syncnet"


@lru_cache(maxsize=1)
def _get_syncnet_pipeline() -> SyncNetPipeline:
    """
    Lazy-load SyncNet once and reuse.
    """
    s3fd_weights = WEIGHTS_DIR / "sfd_face.pth"
    syncnet_weights = WEIGHTS_DIR / "syncnet_v2.model"

    if not s3fd_weights.exists():
        raise FileNotFoundError(f"S3FD weights not found at {s3fd_weights}")
    if not syncnet_weights.exists():
        raise FileNotFoundError(f"SyncNet weights not found at {syncnet_weights}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = SyncNetPipeline(
        s3fd_weights=str(s3fd_weights),
        syncnet_weights=str(syncnet_weights),
        device=device,
    )
    return pipeline


def run_syncnet(
    video_path: Path,
    reference_audio_path: Path | None = None,
) -> tuple[bool, float | None, float | None]:
    """
    Run SyncNet and return (success, best_confidence, best_min_dist).

    reference_audio_path:
      - if provided, SyncNet compares validator-generated reference audio
        against the miner video (preferred).
      - if None, SyncNet extracts audio from the video (fallback).
    """
    pipeline = _get_syncnet_pipeline()

    (
        offset_list,
        confidence_list,
        min_dist_list,
        best_confidence,
        best_min_dist,
        detections_json,
        success,
    ) = pipeline.inference(
        video_path=str(video_path),
        audio_path=str(reference_audio_path) if reference_audio_path else None,
    )

    return bool(success), float(best_confidence), float(best_min_dist)


def compute_syncnet_score(video_path: Path) -> float:
    """
    Use SyncNet to compute an audio-video sync score in [0,1].

    - SyncNetPipeline returns:
        offset_list, confidence_list, min_dist_list,
        best_confidence, best_min_dist, detections_json, success

    We convert:
      - higher best_confidence -> better
      - lower best_min_dist -> better
    into a scalar S_sync in [0,1].
    """
    success, best_confidence, best_min_dist = run_syncnet(video_path)

    if not success or best_confidence is None:
        # No face / no usable crop → treat as badly synced
        return 0.0

    # Heuristic normalization:
    # From the docs: good sync often shows confidence ~4.5-5.5+.
    # Map confidence in [0, 10] -> [0,1]
    conf = float(best_confidence)
    S_conf = max(0.0, min(conf / 10.0, 1.0))

    # min_dist: lower is better. Assume [0, 10] range; map to [1,0] then clamp.
    dist = float(best_min_dist)
    S_dist = 1.0 / (1.0 + max(dist, 0.0))  # 0 -> 1, 1 -> 0.5, 4 -> 0.2, etc.

    # Combine both; tune weights later with real data
    S_sync = 0.9 * S_conf + 0.1 * S_dist
    return float(max(0.0, min(S_sync, 1.0)))


# if __name__ == "__main__":
#     print(compute_lip_sync_score(Path("test_data/talker.mp4")))