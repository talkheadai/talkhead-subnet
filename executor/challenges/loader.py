from __future__ import annotations

import os
import random
from typing import Any

import httpx
from loguru import logger

from executor.challenges.freesound_search import (
    FREESOUND_SEARCH_URL,
    FREESOUND_SOUND_URL,
    audio_label,
    freesound_sort,
    is_acceptable_speech_hit,
    parse_vetted_sound_ids,
    preview_url,
    search_page,
    search_requests,
)
from executor.models import Challenge

DEFAULT_CHALLENGE_COUNT = 7


class ChallengeLoader:
    """
    Build evaluation challenges by pairing frontal portrait photos (Pexels)
    with filtered human speech clips from the Freesound API.
    """

    def __init__(
        self,
        *,
        pexels_api_key: str | None = None,
        freesound_api_key: str | None = None,
        challenge_count: int | None = None,
        timeout_sec: float = 60.0,
    ) -> None:
        self._pexels_api_key = (pexels_api_key or os.getenv("PEXELS_API_KEY", "")).strip()
        self._freesound_api_key = (
            freesound_api_key or os.getenv("FREESOUND_API_KEY", "")
        ).strip()
        self._challenge_count = max(
            1,
            challenge_count
            or int(os.getenv("CHALLENGE_COUNT", str(DEFAULT_CHALLENGE_COUNT))),
        )
        self._timeout_sec = timeout_sec

    def load(self) -> list[Challenge]:
        if not self._pexels_api_key:
            raise ValueError("PEXELS_API_KEY is required to load challenges")
        if not self._freesound_api_key:
            raise ValueError("FREESOUND_API_KEY is required to load challenges")

        face_images = self._fetch_face_images(self._challenge_count)
        audio_clips = self._fetch_speech_clips(self._challenge_count)
        challenges: list[Challenge] = []

        for idx, (face_bytes, clip) in enumerate(zip(face_images, audio_clips)):
            challenges.append(
                Challenge(
                    challenge_id=f"challenge-{idx}-{clip['id']}",
                    text=clip["text"],
                    audio_bytes=clip["audio_bytes"],
                    face_bytes=face_bytes,
                )
            )

        logger.info(f"loaded {len(challenges)} challenges from Pexels + Freesound")
        return challenges

    def _freesound_headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {self._freesound_api_key}"}

    def _fetch_face_images(self, count: int) -> list[bytes]:
        queries = ("portrait face frontal", "headshot portrait", "close up face portrait")
        photos: list[dict] = []

        with httpx.Client(timeout=self._timeout_sec) as client:
            for query in queries:
                if len(photos) >= count:
                    break
                response = client.get(
                    "https://api.pexels.com/v1/search",
                    params={
                        "query": query,
                        "per_page": max(count, 15),
                        "orientation": "portrait",
                    },
                    headers={"Authorization": self._pexels_api_key},
                )
                response.raise_for_status()
                payload = response.json()
                for photo in payload.get("photos", []):
                    if isinstance(photo, dict):
                        photos.append(photo)

        if len(photos) < count:
            raise ValueError(
                f"Pexels returned {len(photos)} portrait photos, need {count}"
            )

        random.shuffle(photos)
        selected = photos[:count]
        images: list[bytes] = []

        with httpx.Client(timeout=self._timeout_sec, follow_redirects=True) as client:
            for photo in selected:
                src = photo.get("src") if isinstance(photo.get("src"), dict) else {}
                image_url = (
                    src.get("large")
                    or src.get("medium")
                    or src.get("original")
                    or photo.get("url")
                )
                if not isinstance(image_url, str) or not image_url:
                    raise ValueError("Pexels photo missing downloadable URL")
                image_response = client.get(image_url)
                image_response.raise_for_status()
                images.append(image_response.content)

        return images

    def _fetch_speech_clips(self, count: int) -> list[dict[str, Any]]:
        vetted_ids = parse_vetted_sound_ids()
        if vetted_ids:
            return self._fetch_vetted_clips(vetted_ids, count)
        return self._fetch_filtered_search_clips(count)

    def _fetch_vetted_clips(
        self, sound_ids: list[int], count: int
    ) -> list[dict[str, Any]]:
        random.shuffle(sound_ids)
        selected_ids = sound_ids[:count]
        if len(selected_ids) < count:
            raise ValueError(
                f"FREESOUND_SOUND_IDS has {len(sound_ids)} ids, need {count}"
            )

        clips: list[dict[str, Any]] = []
        with httpx.Client(timeout=self._timeout_sec) as client:
            for sound_id in selected_ids:
                response = client.get(
                    FREESOUND_SOUND_URL.format(sound_id=sound_id),
                    params={
                        "token": self._freesound_api_key,
                        "fields": "id,name,tags,previews,duration,avg_rating",
                    },
                    headers=self._freesound_headers(),
                )
                response.raise_for_status()
                hit = response.json()
                clip = self._download_clip(client, hit)
                if clip is not None:
                    clips.append(clip)

        if len(clips) < count:
            raise ValueError(
                f"Only downloaded {len(clips)} vetted Freesound clips, need {count}"
            )
        return clips

    def _fetch_filtered_search_clips(self, count: int) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        target_pool = max(count * 3, 21)

        with httpx.Client(timeout=self._timeout_sec) as client:
            for query, filter_expr in search_requests():
                if len(hits) >= target_pool:
                    break
                params = {
                    "query": query,
                    "token": self._freesound_api_key,
                    "fields": "id,name,tags,previews,duration,avg_rating",
                    "page": search_page(),
                    "page_size": min(150, target_pool),
                    "sort": freesound_sort(),
                    "filter": filter_expr,
                }
                response = client.get(
                    FREESOUND_SEARCH_URL,
                    params=params,
                    headers=self._freesound_headers(),
                )
                if response.status_code == 404:
                    logger.warning(
                        f"Freesound search returned no results for query={query!r}"
                    )
                    continue
                response.raise_for_status()
                payload = response.json()
                for hit in payload.get("results", []):
                    if not isinstance(hit, dict):
                        continue
                    hit_id = hit.get("id")
                    if not isinstance(hit_id, int) or hit_id in seen_ids:
                        continue
                    if not is_acceptable_speech_hit(hit):
                        continue
                    seen_ids.add(hit_id)
                    hits.append(hit)

        if len(hits) < count:
            raise ValueError(
                f"Freesound returned {len(hits)} acceptable speech clips, need {count}"
            )

        random.shuffle(hits)
        selected = hits[:count]
        clips: list[dict[str, Any]] = []

        with httpx.Client(timeout=self._timeout_sec, follow_redirects=True) as client:
            for hit in selected:
                clip = self._download_clip(client, hit)
                if clip is not None:
                    clips.append(clip)

        if len(clips) < count:
            raise ValueError(
                f"Only downloaded {len(clips)} Freesound clips, need {count}"
            )
        return clips

    def _download_clip(
        self, client: httpx.Client, hit: dict[str, Any]
    ) -> dict[str, Any] | None:
        url = preview_url(hit)
        if not url:
            logger.warning(f"Freesound sound {hit.get('id')} missing preview URL")
            return None
        audio_response = client.get(url)
        audio_response.raise_for_status()
        return {
            "id": hit.get("id"),
            "text": audio_label(hit),
            "audio_bytes": audio_response.content,
        }
