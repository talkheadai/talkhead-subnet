from .media_utils import extract_audio
import soundfile as sf
import numpy as np
from pathlib import Path
import cv2

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

def _compute_motion_and_freeze(frames: list[np.ndarray]) -> tuple[float, float]:
    """
    Compute:
      - avg_motion: average frame-to-frame difference (0-255)
      - freeze_ratio: fraction of pairs that are 'almost identical'
    """
    if len(frames) < 2:
        return 0.0, 1.0  # no frames -> "frozen" and no motion

    diffs = []
    frozen_pairs = 0
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        mean_diff = float(diff.mean())  # 0-255
        diffs.append(mean_diff)

        # if almost no pixel change, treat pair as frozen
        if mean_diff < 2.0:  # threshold; tune later
            frozen_pairs += 1

        prev_gray = gray

    if not diffs:
        return 0.0, 1.0

    avg_motion = float(np.mean(diffs))
    freeze_ratio = frozen_pairs / len(diffs)
    return avg_motion, freeze_ratio

def score_audio_quality(video_path: Path) -> float:
    """
    Audio quality score in [0,1], combining:
      - loudness (RMS in a reasonable range)
      - silence fraction (penalize if mostly silent)
      - clipping fraction (penalize if heavily clipped)
    """
    audio_path = extract_audio(video_path)
    if audio_path is None:
        return 0.0

    try:
        audio, sr = sf.read(str(audio_path))
    except Exception:
        return 0.0

    # Ensure mono float32 in [-1,1]
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if audio.size == 0:
        return 0.0

    # Overall RMS
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms <= 1e-8:
        return 0.0

    # dBFS (0 dBFS is max amplitude)
    rms_db = 20.0 * np.log10(rms + 1e-9)

    # Map loudness:
    #  - target band: roughly [-25, -12] dBFS
    #  - < -40 or > -3: very bad
    if rms_db <= -40.0 or rms_db >= -3.0:
        S_loud = 0.0
    elif -25.0 <= rms_db <= -12.0:
        S_loud = 1.0
    elif rms_db > -25.0:
        # ramp from -25 -> -12
        S_loud = (rms_db - (-25.0)) / (-12.0 - (-25.0))
    else:  # -40 < rms_db < -25
        S_loud = (rms_db - (-40.0)) / (-25.0 - (-40.0))

    # Frame-based RMS for silence fraction
    frame_len = int(sr * 0.05)  # 50ms
    if frame_len <= 0:
        return max(0.0, min(S_loud, 1.0))

    num_frames = audio.size // frame_len
    if num_frames <= 0:
        return max(0.0, min(S_loud, 1.0))

    frames = audio[: num_frames * frame_len].reshape(num_frames, frame_len)
    frame_rms = np.sqrt(np.mean(frames**2, axis=1))
    silence_threshold = 10 ** (-40.0 / 20.0)  # -40 dBFS
    silent_frames = np.sum(frame_rms < silence_threshold)
    silence_frac = silent_frames / num_frames

    # Allow some silence (pauses), but penalize if > 50%
    if silence_frac <= 0.1:
        S_silence = 1.0
    elif silence_frac >= 0.5:
        S_silence = 0.0
    else:
        # linear from 1 at 0.1 to 0 at 0.5
        S_silence = 1.0 - (silence_frac - 0.1) / (0.5 - 0.1)

    # Clipping fraction: |x| close to 1
    clipping_thresh = 0.99
    clipping_frac = float(np.mean(np.abs(audio) > clipping_thresh))

    if clipping_frac <= 0.001:
        S_clip = 1.0
    elif clipping_frac >= 0.05:
        S_clip = 0.0
    else:
        # linear from 1 at 0.1% to 0 at 5%
        S_clip = 1.0 - (clipping_frac - 0.001) / (0.05 - 0.001)

    # Combine audio metrics
    S_audio = 0.5 * S_loud + 0.3 * S_silence + 0.2 * S_clip
    return float(np.clip(S_audio, 0.0, 1.0))
