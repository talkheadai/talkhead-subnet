from __future__ import annotations

import os
import tempfile
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from piper.voice import PiperVoice

from utils.media import extract_audio


PIPER_VOICE_DIR = Path(
    os.getenv(
        "PIPER_VOICE_DIR",
        Path(__file__).resolve().parents[1] / "piper-voices",
    )
)


@lru_cache(maxsize=8)
def _load_voice(voice_profile: str) -> PiperVoice:
    """
    Load a Piper voice by searching for `<voice_profile>.onnx` under PIPER_VOICE_DIR.
    """
    matches = list(PIPER_VOICE_DIR.rglob(f"{voice_profile}.onnx"))
    if not matches:
        raise FileNotFoundError(
            f"Voice profile '{voice_profile}' not found under {PIPER_VOICE_DIR}"
        )
    return PiperVoice.load(str(matches[0]))


def _synthesize_audio(text: str, voice_profile: str) -> Path:
    voice = _load_voice(voice_profile)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp, "wb") as f:
        voice.synthesize_wav(text, f)
    return Path(tmp.name)


def _resample_linear(audio: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    if src_sr == tgt_sr or audio.size == 0:
        return audio
    duration = audio.shape[0] / float(src_sr)
    tgt_len = max(1, int(duration * tgt_sr))
    x_old = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    x_new = np.linspace(0.0, duration, num=tgt_len, endpoint=False)
    resampled = np.interp(x_new, x_old, audio).astype(np.float32)
    return resampled


def _load_audio(path: Path, target_sr: int = 16_000) -> np.ndarray:
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    audio = _resample_linear(audio, sr, target_sr)
    return audio


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    L = min(a.shape[0], b.shape[0])
    a = a[:L]
    b = b[:L]
    a = a - float(a.mean())
    b = b - float(b.mean())
    a_norm = float(np.linalg.norm(a) + 1e-8)
    b_norm = float(np.linalg.norm(b) + 1e-8)
    a = a / a_norm
    b = b / b_norm
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def verify_video_audio_matches_text(
    text: str,
    voice_profile: str,
    video_path: Path,
    similarity_threshold: float = 0.85,
) -> tuple[bool, str]:
    """
    Generate audio from text + voice_profile (Piper) and compare to the video's audio.
    Returns (ok, reason).
    """
    video_audio_path = extract_audio(video_path)
    if video_audio_path is None:
        return False, "failed to extract audio from video"

    try:
        ref_audio_path = _synthesize_audio(text, voice_profile)
    except Exception as exc:  # noqa: BLE001
        return False, f"failed to synthesize reference audio: {exc}"

    try:
        video_audio = _load_audio(video_audio_path)
        ref_audio = _load_audio(ref_audio_path)
        sim = _cosine_sim(video_audio, ref_audio)
    finally:
        try:
            ref_audio_path.unlink(missing_ok=True)
        except Exception:
            pass

    if sim >= similarity_threshold:
        return True, ""
    return False, f"incorrect audio (similarity={sim:.3f} < {similarity_threshold})"
