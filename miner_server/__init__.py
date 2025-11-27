"""Public API surface for the TalkHead miner server package."""

from .generate import generate_talking_head
from .sadtalker_backend import generate_video_with_sadtalker

__all__ = ["generate_talking_head", "generate_video_with_sadtalker"]
__version__ = "0.1.0"
