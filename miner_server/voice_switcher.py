import torch
from piper.voice import PiperVoice
import os
import wave
import tempfile
from pathlib import Path

# piper voices directory is in parent directory of this file
PIPER_VOICE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "piper-voices")
print(PIPER_VOICE_DIR)



class VoiceSwitcher:
    def __init__(self, piper_voice_dir: str = PIPER_VOICE_DIR):
        self.piper_voice_dir = piper_voice_dir

        self.voices = {}
        # Load voices (download ONNX files to ./voices/) iterating all files in the piper_voice_dir deeply
        for root, dirs, files in os.walk(piper_voice_dir):
            for file in files:
                if os.path.join(root, file).endswith(".onnx") and file.startswith("en"):
                    voice_name = os.path.splitext(file)[0]
                    print(voice_name)
                    self.voices[voice_name] = PiperVoice.load(os.path.join(root, file)) 
                    if len(self.voices) == 3:
                        return

    def generate_audio(self, text: str, voice_profile: str = "en_US-lessac-medium"):
        if voice_profile not in self.voices:
            raise ValueError(f"Voice {voice_profile} not found")
        voice = self.voices[voice_profile]
        output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(output_path, "wb") as f:
            voice.synthesize_wav(text, f)
        return Path(output_path.name)

# === Usage Example ===
if __name__ == "__main__":
    switcher = VoiceSwitcher(piper_voice_dir="../piper-voices")
    
    # Generate with different stock voices
    audio_path = switcher.generate_audio("Hello, welcome to TalkHead!", "en_GB-alan-low")
    print(audio_path)