from pathlib import Path
from functools import lru_cache
from typing import Optional

import whisper


@lru_cache(maxsize=1)
def _get_model(model_name: str = "small") -> whisper.Whisper:
    # Adjust model size depending on your GPU
    return whisper.load_model(model_name)


def transcribe_audio(
    audio_path: Path,
    language: Optional[str] = None,
    model_name: str = "small",
) -> str:
    """
    Transcribe audio using Whisper and return lowercased text.
    """
    model = _get_model(model_name)
    kwargs = {}
    if language:
        # language like "en-US" -> "en"
        kwargs["language"] = language.split("-")[0]
    result = model.transcribe(str(audio_path), **kwargs)
    text = result.get("text", "")
    return text.strip().lower()
