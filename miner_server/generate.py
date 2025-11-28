import tempfile
from pathlib import Path
from typing import Optional

from gtts import gTTS

from sadtalker_backend import generate_video_with_sadtalker


def _synthesize_speech_gtts(text: str, language: str = "en") -> Path:
    """
    TTS using gTTS into a temporary .wav or .mp3 file.
    SadTalker usually supports .wav, but check docs; here we use .wav.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    tts = gTTS(text=text, lang=language)
    tts.save(tmp.name)
    return Path(tmp.name)


def generate_talking_head(
    image_bytes: bytes,
    text: str,
    language: str = "en",
    duration_sec: Optional[float] = None,
) -> bytes:
    """
    New generation path:

      1. Text -> TTS (gTTS) -> audio.wav
      2. Image bytes + audio.wav -> SadTalker -> talking-head MP4

    The outer API (miner HTTP API) stays the same.
    """
    # 1. TTS
    audio_path = _synthesize_speech_gtts(text, language=language)

    # 2. SadTalker
    video_bytes = generate_video_with_sadtalker(
        image_bytes=image_bytes,
        audio_path=audio_path,
        duration_sec=duration_sec,
    )

    return video_bytes
