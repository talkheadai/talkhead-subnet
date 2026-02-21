import torch
from piper.voice import PiperVoice
import os
import wave
import tempfile
from pathlib import Path
# piper voices directory is in parent directory of this file
PIPER_VOICE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "piper-voices")



class VoiceSwitcher:
    def __init__(self, piper_voice_dir: str = PIPER_VOICE_DIR):
        self.piper_voice_dir = piper_voice_dir

        self.voices = {}
        self.voice_paths = {}
        # Index voices without loading models (lazy load on demand).
        for root, dirs, files in os.walk(piper_voice_dir):
            for file in files:
                if os.path.join(root, file).endswith(".onnx") and file.startswith("en"):
                    voice_name = os.path.splitext(file)[0]
                    self.voice_paths[voice_name] = os.path.join(root, file)

    def generate_audio(self, text: str, voice_profile: str = "en_US-lessac-medium"):
        voice = self.voices.get(voice_profile)
        if voice is None:
            voice_path = self.voice_paths.get(voice_profile)
            if voice_path is None:
                raise ValueError(f"Voice {voice_profile} not found")
            voice = PiperVoice.load(voice_path)
            self.voices[voice_profile] = voice
        output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(output_path, "wb") as f:
            voice.synthesize_wav(text, f)
        return Path(output_path.name)

# === Usage Example ===
if __name__ == "__main__":
    switcher = VoiceSwitcher(piper_voice_dir="../piper-voices")
    
    # Generate with different stock voices
    switcher.generate_audio("Hello, welcome to TalkHead!", "en_GB-alan-low")