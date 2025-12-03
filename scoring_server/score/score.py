import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import soundfile as sf  # NEW
from utils.media import probe_duration, extract_audio
from utils.asr import transcribe_audio
from .quality import score_audio_quality, _sample_frames, _compute_motion_and_freeze
from score.lipsync import compute_lip_sync_score
from score.faceid import compute_face_identity_score
from score.metric_weights import metric_weights

@dataclass
class MinerEvalInput:
    miner_id: str
    text: str
    language: str
    latency_sec: float
    video_path: Path
    target_duration_sec: float = 8.0
    ref_face_path: Path | None = None     # <-- add this


@dataclass
class MinerEvalScores:
    miner_id: str
    S_text: float
    # S_duration: float # TODO: add this back in
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

def score_text(text: str, language: str, video_path: Path) -> float:
    audio_path = extract_audio(video_path)
    if audio_path is None:
        return 0.0

    recognized = transcribe_audio(audio_path, language=language)
    if not recognized:
        return 0.0

    ref = text.strip().lower()
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
        return min(1.0, 1 - 0.3 * abs(target_duration_sec - dur))

    # penalize > max_sec
    return max(0.0, 1.0 - (dur - max_sec) / max_sec)


def score_latency(latency_sec: float, cap_sec: float = 100.0, duration_sec: float = 8.0) -> float:
    return max(0.0, 1.0 - (latency_sec / cap_sec) * (8.0 / duration_sec))


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
      - visual sharpness (Laplacian variance)
      - brightness (mean intensity)
      - motion amount (some movement is good)
      - freeze ratio (long static segments are bad)
      - audio quality (loudness, silence, clipping)
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

    avg_sharp = float(np.mean(sharp_vals))
    avg_bright = float(np.mean(bright_vals))

    # Map sharpness to [0,1]
    #  - below 50: very blurry -> 0
    #  - 50-300: ramp to 1
    #  - >300: cap at 1
    if avg_sharp <= 50.0:
        S_sharp = 0.0
    elif avg_sharp >= 300.0:
        S_sharp = 1.0
    else:
        S_sharp = (avg_sharp - 50.0) / (300.0 - 50.0)

    # Map brightness to [0,1], prefer roughly [60, 200]
    if avg_bright <= 20.0 or avg_bright >= 235.0:
        S_bright = 0.0
    elif 60.0 <= avg_bright <= 200.0:
        S_bright = 1.0
    elif avg_bright < 60.0:
        S_bright = (avg_bright - 20.0) / (60.0 - 20.0)
    else:  # > 200
        S_bright = (235.0 - avg_bright) / (235.0 - 200.0)

    # Motion + freeze
    avg_motion, freeze_ratio = _compute_motion_and_freeze(frames)
    # Map motion [0, 20] -> [0,1], clamp
    S_motion = float(np.clip(avg_motion / 20.0, 0.0, 1.0))
    # Freeze score: 1 when no frozen pairs, 0 when all pairs frozen
    S_freeze = float(np.clip(1.0 - freeze_ratio, 0.0, 1.0))

    # Audio quality
    S_audio = score_audio_quality(video_path)

    # Combine: tweak weights as you like
    S_quality = (
        0.3 * S_sharp    # detail / blur
        + 0.15 * S_bright  # not too dark/bright
        + 0.15 * S_motion  # some movement
        + 0.15 * S_freeze  # no long freezes
        + 0.25 * S_audio   # sound is important
    )

    return float(np.clip(S_quality, 0.0, 1.0))



def evaluate_miner(e: MinerEvalInput) -> MinerEvalScores:
    S_text = score_text(e.text, e.language, e.video_path)
    S_duration = score_duration(e.video_path, e.target_duration_sec)
    S_latency = score_latency(e.latency_sec, duration_sec=probe_duration(e.video_path))
    S_sync = score_sync(e.video_path)
    S_face = score_face(e.video_path, e.ref_face_path)
    S_quality = score_quality(e.video_path)


    composite = (
        metric_weights["text"] * S_text
        + metric_weights["duration"] * S_duration
        + metric_weights["latency"] * S_latency
        + metric_weights["sync"] * S_sync
        + metric_weights["face"] * S_face
        + metric_weights["quality"] * S_quality
    )

    reason = (
        f"S_text={S_text:.2f}, "
        # f"S_duration={S_duration:.2f}, "
        f"S_latency={S_latency:.2f}, "
        f"S_sync={S_sync:.2f}, "
        f"S_face={S_face:.2f}, "
        f"S_quality={S_quality:.2f}"
    )

    return MinerEvalScores(
        miner_id=e.miner_id,
        S_text=S_text,
        # S_duration=S_duration,
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
