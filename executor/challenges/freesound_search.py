from __future__ import annotations

import os
from typing import Any

FREESOUND_SEARCH_URL = "https://freesound.org/apiv2/search/"
FREESOUND_SOUND_URL = "https://freesound.org/apiv2/sounds/{sound_id}/"

DEFAULT_DURATION_FILTER = "duration:[3 TO 10] tag:speech tag:english avg_rating:[4 TO *]"
DEFAULT_SORT = "rating_desc"

SPEECH_SEARCH_QUERIES: tuple[str, ...] = (
    "+talking +human -robot -synthesizer -alien -gibberish -crowd",
    "talking woman",
    "man speaking",
    "dialogue english",
)

TAG_ONLY_FILTERS: tuple[str, ...] = (
    "duration:[3 TO 20] tag:talking tag:english tag:female avg_rating:[4 TO *]",
    "duration:[3 TO 20] tag:talking tag:english tag:male avg_rating:[4 TO *]",
    "duration:[3 TO 20] tag:voice-acting tag:english avg_rating:[4 TO *]",
)

PREVIEW_FIELDS: tuple[str, ...] = (
    "preview-hq-mp3",
    "preview-lq-mp3",
    "preview-hq-ogg",
    "preview-lq-ogg",
)

BLOCKED_TAGS = frozenset(
    {
        "robot",
        "robotic",
        "synthesizer",
        "synthetic",
        "alien",
        "gibberish",
        "crowd",
        "scifi",
        "sci-fi",
        "radio",
        "am",
        "shortwave",
        "babble",
        "babbling",
        "baby",
        "infant",
        "computer",
        "android",
        "mechanize",
        "processed",
    }
)

REQUIRED_SPEECH_TAGS = frozenset(
    {
        "speech",
        "talking",
        "speaking",
        "spoken",
        "talk",
        "dialogue",
        "voice-acting",
        "voiceover",
        "narration",
        "vocal",
    }
)


def freesound_duration_filter() -> str:
    return os.getenv("FREESOUND_DURATION_FILTER", DEFAULT_DURATION_FILTER).strip()


def freesound_sort() -> str:
    return os.getenv("FREESOUND_SORT", DEFAULT_SORT).strip()


def parse_vetted_sound_ids() -> list[int]:
    raw = os.getenv("FREESOUND_SOUND_IDS", "").strip()
    if not raw:
        return []
    ids: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError:
            continue
    return ids


def normalize_tags(hit: dict[str, Any]) -> set[str]:
    tags = hit.get("tags")
    if isinstance(tags, str):
        return {tag.strip().lower() for tag in tags.split(",") if tag.strip()}
    if isinstance(tags, list):
        return {str(tag).strip().lower() for tag in tags if str(tag).strip()}
    return set()


def is_acceptable_speech_hit(hit: dict[str, Any]) -> bool:
    tags = normalize_tags(hit)
    if tags & BLOCKED_TAGS:
        return False
    if not (tags & REQUIRED_SPEECH_TAGS):
        return False
    return preview_url(hit) is not None


def preview_url(hit: dict[str, Any]) -> str | None:
    previews = hit.get("previews")
    if not isinstance(previews, dict):
        return None
    for key in PREVIEW_FIELDS:
        value = previews.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def audio_label(hit: dict[str, Any]) -> str:
    name = hit.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    tags = hit.get("tags")
    if isinstance(tags, str) and tags.strip():
        return tags.strip()
    if isinstance(tags, list) and tags:
        return ", ".join(str(tag) for tag in tags[:5])
    hit_id = hit.get("id")
    return f"freesound-{hit_id}" if hit_id is not None else "freesound-audio"


def search_page() -> int:
    fixed = os.getenv("FREESOUND_PAGE", "").strip()
    if fixed:
        return max(1, int(fixed))
    return 1


def search_requests() -> list[tuple[str, str]]:
    """Return (query, filter) pairs to try in order."""
    base_filter = freesound_duration_filter()
    requests: list[tuple[str, str]] = [(query, base_filter) for query in SPEECH_SEARCH_QUERIES]
    requests.extend(("", tag_filter) for tag_filter in TAG_ONLY_FILTERS)
    return requests
