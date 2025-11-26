import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .media_utils import probe_duration, extract_audio
from .asr import transcribe_audio
from .lipsync_backend import compute_lip_sync_score
from .faceid_backend import compute_face_identity_score

@dataclass
class MinerEvalInput:
    miner_id: str
    script: str
    language: str
    latency_ms: float
    video_path: Path
    target_duration_sec: float = 8.0
    ref_face_path: Path | None = None     # <-- add this


@dataclass
class MinerEvalScores:
    miner_id: str
    S_script: float
    S_duration: float
    S_latency: float
    S_sync: float
    S_face: float
    S_quality: float
    composite: float
    reason: str


def _text_similarity(a: str, b: str) -> float:
    """
    Simple normalized similarity in [0,1] using difflib.
    """
    return difflib.SequenceMatcher(None, a, b).ratio()


def score_script(script: str, language: str, video_path: Path) -> float:
    audio_path = extract_audio(video_path)
    if audio_path is None:
        return 0.0

    recognized = transcribe_audio(audio_path, language=language)
    if not recognized:
        return 0.0

    ref = script.strip().lower()
    return _text_similarity(ref, recognized)


def score_duration(
    video_path: Path,
    target_duration_sec: float,
    min_sec: float = 3.0,
    max_sec: float = 15.0,
) -> float:
    dur = probe_duration(video_path)
    if dur is None or dur <= 0:
        return 0.0

    # Hard reject extremely off durations
    if dur < 0.5:
        return 0.0

    # Prefer something within [min_sec, max_sec]
    if dur < min_sec:
        # linear ramp from 0 at 0.5s to 1 at min_sec
        return max(0.0, (dur - 0.5) / (min_sec - 0.5))
    if dur <= max_sec:
        return 1.0

    # penalize > max_sec
    return max(0.0, 1.0 - (dur - max_sec) / max_sec)


def score_latency(latency_ms: float, cap_ms: float = 10_000.0) -> float:
    return max(0.0, 1.0 - latency_ms / cap_ms)

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

def score_sync(video_path: Path) -> float:
    """
    Lip-sync score using SyncNet:
      0.0 ~ badly out-of-sync / unusable
      1.0 ~ very well-synced according to SyncNet
    """
    try:
        return compute_lip_sync_score(video_path)
    except Exception as e:
        # Don't crash scoring if SyncNet fails on a weird video; just downscore.
        # You can also log `e` here.
        return 0.0

def score_face(video_path: Path, ref_face_path: Optional[Path]) -> float:
    """
    Face identity consistency:
      - if ref_face_path is provided: use embedding-based identity check
      - else: fallback to simple 'face presence' via Haar (or 0.5).
    """
    if ref_face_path is None:
        # fallback: we don't know the reference → can't judge identity
        # you can replace this with simple presence score if you want
        return 0.5

    try:
        return compute_face_identity_score(ref_face_path, video_path)
    except Exception:
        # don't kill scoring if identity model fails
        return 0.0

def score_quality(video_path: Path) -> float:
    """
    Basic perceptual quality score in [0,1], combining:
    - sharpness (Laplacian variance)
    - brightness (mean intensity)
    - motion (reusing motion index)
    """
    frames = _sample_frames(video_path, max_frames=32)
    if not frames:
        return 0.0

    sharp_vals = []
    bright_vals = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Sharpness via Laplacian variance
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(lap.var())
        sharp_vals.append(sharpness)

        # Brightness via mean pixel value
        brightness = float(gray.mean())
        bright_vals.append(brightness)

    # Aggregate
    avg_sharp = float(np.mean(sharp_vals))
    avg_bright = float(np.mean(bright_vals))

    # Map sharpness to [0,1] with heuristic thresholds
    #  - below 50: very blurry -> 0
    #  - 50-300: ramp up to 1
    #  - >300: cap at 1
    if avg_sharp <= 50:
        S_sharp = 0.0
    elif avg_sharp >= 300:
        S_sharp = 1.0
    else:
        S_sharp = (avg_sharp - 50.0) / (300.0 - 50.0)

    # Map brightness to [0,1], prefer roughly [60, 200]
    if avg_bright <= 20 or avg_bright >= 235:
        S_bright = 0.0
    elif 60 <= avg_bright <= 200:
        S_bright = 1.0
    elif avg_bright < 60:
        S_bright = (avg_bright - 20.0) / (60.0 - 20.0)
    else:  # >200
        S_bright = (235.0 - avg_bright) / (235.0 - 200.0)

    # Motion index from score_sync (reuse so we don’t duplicate work too much)
    S_motion = score_sync(video_path)

    # Combine: weight sharpness + brightness + motion
    S_quality = 0.4 * S_sharp + 0.3 * S_bright + 0.3 * S_motion
    return float(np.clip(S_quality, 0.0, 1.0))



def evaluate_miner(e: MinerEvalInput) -> MinerEvalScores:
    S_script = score_script(e.script, e.language, e.video_path)
    S_duration = score_duration(e.video_path, e.target_duration_sec)
    S_latency = score_latency(e.latency_ms)

    # stubs for now
    S_sync = score_sync(e.video_path)
    S_face = score_face(e.video_path, e.ref_face_path)
    S_quality = score_quality(e.video_path)

    # weights (v1)
    w_script = 0.40
    w_duration = 0.20
    w_latency = 0.10
    w_sync = 0.10
    w_face = 0.10
    w_quality = 0.10

    composite = (
        w_script * S_script
        + w_duration * S_duration
        + w_latency * S_latency
        + w_sync * S_sync
        + w_face * S_face
        + w_quality * S_quality
    )

    reason = (
        f"S_script={S_script:.2f}, "
        f"S_duration={S_duration:.2f}, "
        f"S_latency={S_latency:.2f}, "
        f"S_sync={S_sync:.2f}, "
        f"S_face={S_face:.2f}, "
        f"S_quality={S_quality:.2f}"
    )

    return MinerEvalScores(
        miner_id=e.miner_id,
        S_script=S_script,
        S_duration=S_duration,
        S_latency=S_latency,
        S_sync=S_sync,
        S_face=S_face,
        S_quality=S_quality,
        composite=composite,
        reason=reason,
    )


def normalize_composite(scores: list[MinerEvalScores]) -> dict[str, float]:
    vals = [s.composite for s in scores if s.composite > 0]
    if not vals:
        return {s.miner_id: 0.0 for s in scores}

    max_val = max(vals)
    if max_val <= 0:
        return {s.miner_id: 0.0 for s in scores}

    return {s.miner_id: s.composite / max_val for s in scores}
