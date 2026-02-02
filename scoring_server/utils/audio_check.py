from __future__ import annotations

import os
import tempfile
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from piper.voice import PiperVoice
import librosa

from utils.media import extract_audio
from voice_switcher import VoiceSwitcher
switcher = VoiceSwitcher()


PIPER_VOICE_DIR = Path(
    os.getenv(
        "PIPER_VOICE_DIR",
        Path(__file__).resolve().parents[1] / "piper-voices",
    )
)


def trim_silence(audio: np.ndarray, top_db=30) -> np.ndarray:
    """Remove trailing silence (very common in TTS outputs)"""
    return librosa.effects.trim(audio, top_db=top_db)[0]

def load_mel(path: str, sr=16000, n_mels=80, hop_length=256, n_fft=1024):
    y, _ = librosa.load(path, sr=sr, mono=True)
    y = trim_silence(y)  # ← the key fix
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft,
        fmin=0,
        fmax=sr/2,
        power=1.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.T  # (time_frames, n_mels)

def cosine_sim_mel(a_path: str, b_path: str) -> float:
    mel_a = load_mel(a_path)
    mel_b = load_mel(b_path)

    # Pad the shorter one with zeros (silence) instead of truncating
    if mel_a.shape[0] < mel_b.shape[0]:
        pad = np.zeros((mel_b.shape[0] - mel_a.shape[0], mel_a.shape[1]))
        mel_a = np.vstack([mel_a, pad])
    elif mel_b.shape[0] < mel_a.shape[0]:
        pad = np.zeros((mel_a.shape[0] - mel_b.shape[0], mel_b.shape[1]))
        mel_b = np.vstack([mel_b, pad])

    # Frame-wise cosine similarity (more stable than flattening the whole thing)
    cos_sim_per_frame = np.sum(mel_a * mel_b, axis=1) / (
        np.linalg.norm(mel_a, axis=1) * np.linalg.norm(mel_b, axis=1) + 1e-8
    )
    # Average over frames, ignore silent frames
    valid_frames = np.logical_and(np.linalg.norm(mel_a, axis=1) > 1e-5,
                                  np.linalg.norm(mel_b, axis=1) > 1e-5)
    if np.sum(valid_frames) == 0:
        return 0.0
    return float(np.mean(cos_sim_per_frame[valid_frames]))


def verify_video_audio_matches_text(
    text: str,
    voice_profile: str,
    video_path: Path,
    similarity_threshold: float = 0.96,
) -> tuple[bool, str]:
    """
    Generate audio from text + voice_profile (Piper) and compare to the video's audio.
    Returns (ok, reason).
    """
    video_audio_path = extract_audio(video_path)
    if video_audio_path is None:
        return False, "failed to extract audio from video"

    try:
        ref_audio_path = switcher.generate_audio(text, voice_profile)
    except Exception as exc:  # noqa: BLE001
        return False, f"failed to synthesize reference audio: {exc}"

    try:
        sim = cosine_sim_mel(video_audio_path, ref_audio_path)
    finally:
        try:
            # ref_audio_path.unlink(missing_ok=True)
            pass
        except Exception:
            pass

    if sim >= similarity_threshold:
        return True, ""
    return False, f"incorrect audio (similarity={sim:.3f} < {similarity_threshold})"
