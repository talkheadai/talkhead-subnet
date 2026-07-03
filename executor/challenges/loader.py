from __future__ import annotations

import os

from loguru import logger

from executor.challenges.celebahq import pick_random_face_images
from executor.challenges.librispeech import pick_random_audio_clips
from executor.models import Challenge

DEFAULT_CHALLENGE_COUNT = 7


class ChallengeLoader:
    """
    Build evaluation challenges by pairing random CelebAHQ face images
    with random LibriSpeech clean/train.100 audio clips.
    """

    def __init__(
        self,
        *,
        challenge_count: int | None = None,
    ) -> None:
        self._challenge_count = max(
            1,
            challenge_count
            or int(os.getenv("CHALLENGE_COUNT", str(DEFAULT_CHALLENGE_COUNT))),
        )

    def load(self) -> list[Challenge]:
        face_images = pick_random_face_images(self._challenge_count)
        audio_clips = pick_random_audio_clips(self._challenge_count)
        challenges: list[Challenge] = []

        for idx, (face_bytes, clip) in enumerate(zip(face_images, audio_clips)):
            challenges.append(
                Challenge(
                    challenge_id=f"challenge-{idx}-{clip['id']}",
                    audio_bytes=clip["audio_bytes"],
                    face_bytes=face_bytes,
                )
            )

        logger.info(f"loaded {len(challenges)} challenges from CelebAHQ + LibriSpeech")
        return challenges
