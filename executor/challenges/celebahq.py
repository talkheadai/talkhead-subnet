from __future__ import annotations

import json
import os
import random
from pathlib import Path

from loguru import logger

CELEBAHQ_DATASET_ID = "edgarcancinoe/celebahq_512_id_clusters"
DEFAULT_CELEBAHQ_DATA_DIR = "validator-data/celebahq"

_FACE_PATHS_CACHE: dict[Path, list[Path]] = {}


def celebahq_data_dir() -> Path:
    return Path(os.getenv("CELEBAHQ_DATA_DIR", DEFAULT_CELEBAHQ_DATA_DIR)).expanduser()


def celebahq_faces_dir() -> Path:
    return celebahq_data_dir() / "faces"


def _ready_marker_path() -> Path:
    return celebahq_data_dir() / "ready.json"


def _load_ready_marker() -> dict | None:
    marker = _ready_marker_path()
    if not marker.is_file():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def is_celebahq_dataset_ready() -> bool:
    faces_dir = celebahq_faces_dir()
    marker = _load_ready_marker()
    if marker is None:
        return False
    if marker.get("dataset_id") != CELEBAHQ_DATASET_ID:
        return False
    expected = int(marker.get("image_count", 0) or 0)
    if expected <= 0:
        return False
    existing = sum(1 for _ in faces_dir.glob("*.png"))
    return existing >= expected


def ensure_celebahq_dataset() -> Path:
    """
    Download the CelebAHQ face dataset from Hugging Face and extract PNGs locally.
    Skips work when a previous successful extraction is detected.
    """
    faces_dir = celebahq_faces_dir()
    if is_celebahq_dataset_ready():
        existing = sum(1 for _ in faces_dir.glob("*.png"))
        logger.info(f"celebahq dataset ready: {existing} faces at {faces_dir}")
        return faces_dir

    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")

    from datasets import load_dataset

    root = celebahq_data_dir()
    root.mkdir(parents=True, exist_ok=True)
    faces_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"downloading celebahq dataset: {CELEBAHQ_DATASET_ID}")
    dataset = load_dataset(CELEBAHQ_DATASET_ID, split="train")
    total = len(dataset)

    extracted = 0
    for idx, row in enumerate(dataset):
        file_name = str(row.get("file_name", "")).strip()
        if not file_name:
            file_name = f"celebahq_{idx:05d}.png"
        if not file_name.lower().endswith(".png"):
            file_name = f"{file_name}.png"

        out_path = faces_dir / file_name
        if not out_path.exists():
            image = row.get("image")
            if image is None:
                raise RuntimeError(f"celebahq row {idx} missing image payload")
            image.save(out_path, format="PNG")

        extracted += 1
        if extracted % 1000 == 0 or extracted == total:
            logger.info(f"extracted {extracted}/{total} celebahq faces")

    marker_payload = {
        "dataset_id": CELEBAHQ_DATASET_ID,
        "image_count": extracted,
    }
    _ready_marker_path().write_text(
        json.dumps(marker_payload, indent=2),
        encoding="utf-8",
    )
    _FACE_PATHS_CACHE.pop(faces_dir, None)
    logger.info(f"celebahq dataset extracted to {faces_dir} ({extracted} faces)")
    return faces_dir


def _list_face_paths(faces_dir: Path) -> list[Path]:
    cached = _FACE_PATHS_CACHE.get(faces_dir)
    if cached is not None:
        return cached
    paths = sorted(faces_dir.glob("*.png"))
    _FACE_PATHS_CACHE[faces_dir] = paths
    return paths


def pick_random_face_images(count: int, *, faces_dir: Path | None = None) -> list[bytes]:
    directory = faces_dir or celebahq_faces_dir()
    if not directory.is_dir():
        raise ValueError(
            f"celebahq faces directory not found: {directory}. "
            "Run ensure_celebahq_dataset() during validator setup."
        )

    paths = _list_face_paths(directory)
    if len(paths) < count:
        raise ValueError(
            f"need {count} celebahq faces, found {len(paths)} in {directory}"
        )

    selected = random.sample(paths, count)
    return [path.read_bytes() for path in selected]
