from __future__ import annotations

import glob
import io
import json
import os
import random
import wave
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from loguru import logger

from executor.challenges.hf import configure_hf_auth, resolve_hf_token

LIBRISPEECH_DATASET_ID = "openslr/librispeech_asr"
LIBRISPEECH_CONFIG = "clean"
LIBRISPEECH_SPLIT = "train.100"
LIBRISPEECH_PARQUET_GLOB = "clean/train.100/*.parquet"
DEFAULT_LIBRISPEECH_DATA_DIR = "validator-data/librispeech"
MIN_CLIP_DURATION_SEC = 3.0
MAX_CLIP_DURATION_SEC = 10.0

_CLIP_INDEX_CACHE: dict[Path, list[dict[str, str]]] = {}


def librispeech_data_dir() -> Path:
    return Path(
        os.getenv("LIBRISPEECH_DATA_DIR", DEFAULT_LIBRISPEECH_DATA_DIR)
    ).expanduser()


def librispeech_audio_dir() -> Path:
    return librispeech_data_dir() / "audio"


def _ready_marker_path() -> Path:
    return librispeech_data_dir() / "ready.json"


def _load_ready_marker() -> dict | None:
    marker = _ready_marker_path()
    if not marker.is_file():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def is_librispeech_dataset_ready() -> bool:
    audio_dir = librispeech_audio_dir()
    marker = _load_ready_marker()
    if marker is None:
        return False
    if marker.get("dataset_id") != LIBRISPEECH_DATASET_ID:
        return False
    if marker.get("split") != LIBRISPEECH_SPLIT:
        return False
    expected = int(marker.get("clip_count", 0) or 0)
    if expected <= 0:
        return False
    existing = sum(1 for _ in audio_dir.glob("*.wav"))
    return existing >= expected


def _download_train_100_snapshot() -> Path:
    snapshot_path = snapshot_download(
        LIBRISPEECH_DATASET_ID,
        repo_type="dataset",
        allow_patterns=[LIBRISPEECH_PARQUET_GLOB],
        token=resolve_hf_token(),
    )
    parquet_files = sorted(glob.glob(f"{snapshot_path}/{LIBRISPEECH_PARQUET_GLOB}"))
    if not parquet_files:
        raise RuntimeError(
            f"no librispeech parquet files found under {snapshot_path}/{LIBRISPEECH_PARQUET_GLOB}"
        )
    return Path(snapshot_path)


def _audio_array_and_sr(audio_obj: object) -> tuple[np.ndarray, int]:
    if isinstance(audio_obj, dict):
        if "array" in audio_obj:
            arr = np.asarray(audio_obj["array"], dtype=np.float32).reshape(-1)
            sr = int(audio_obj.get("sampling_rate", audio_obj.get("sample_rate", 16000)))
            return arr, sr
        if "bytes" in audio_obj and audio_obj["bytes"]:
            import torchaudio  # type: ignore

            wav, sr = torchaudio.load(io.BytesIO(audio_obj["bytes"]))
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            return wav.squeeze(0).cpu().numpy().astype(np.float32), int(sr)

    if hasattr(audio_obj, "get_all_samples"):
        samples = audio_obj.get_all_samples()
        arr = np.asarray(samples.data, dtype=np.float32).reshape(-1)
        sr = int(getattr(samples, "sample_rate", 16000) or 16000)
        return arr, sr

    raise TypeError(f"unsupported audio payload type: {type(audio_obj)}")


def _write_wav(path: Path, samples: np.ndarray, sampling_rate: int) -> None:
    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sampling_rate))
        wf.writeframes(pcm.tobytes())


def _safe_clip_filename(clip_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in clip_id.strip())
    return f"{safe}.wav"


def ensure_librispeech_dataset() -> Path:
    """
    Download LibriSpeech clean/train.100 from Hugging Face and extract WAV clips locally.
    Only utterances between 3-10 seconds are kept to match evaluation constraints.
    """
    audio_dir = librispeech_audio_dir()
    if is_librispeech_dataset_ready():
        existing = sum(1 for _ in audio_dir.glob("*.wav"))
        logger.info(f"librispeech dataset ready: {existing} clips at {audio_dir}")
        return audio_dir

    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    token = configure_hf_auth()

    from datasets import load_dataset

    root = librispeech_data_dir()
    root.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = _download_train_100_snapshot()
    parquet_files = sorted(glob.glob(f"{snapshot_path}/{LIBRISPEECH_PARQUET_GLOB}"))
    logger.info(
        f"loading librispeech split={LIBRISPEECH_SPLIT} from {len(parquet_files)} parquet files"
    )

    dataset = load_dataset(
        "parquet",
        data_files={"train": parquet_files},
        split="train",
        cache_dir=str(root / "parquet-cache"),
        token=token,
    )
    total = len(dataset)

    extracted = 0
    skipped = 0
    for idx, row in enumerate(dataset):
        clip_id = str(row.get("id") or row.get("file") or f"clip-{idx}").strip()
        audio_obj = row.get("audio")
        if audio_obj is None:
            skipped += 1
            continue

        try:
            samples, sampling_rate = _audio_array_and_sr(audio_obj)
        except Exception as exc:
            logger.warning(f"failed to decode librispeech clip={clip_id}: {exc}")
            skipped += 1
            continue

        duration_sec = len(samples) / max(sampling_rate, 1)
        if duration_sec < MIN_CLIP_DURATION_SEC or duration_sec > MAX_CLIP_DURATION_SEC:
            skipped += 1
            continue

        out_path = audio_dir / _safe_clip_filename(clip_id)
        if not out_path.exists():
            _write_wav(out_path, samples, sampling_rate)

        extracted += 1
        if extracted % 1000 == 0 or idx + 1 == total:
            logger.info(
                f"extracted {extracted} librispeech clips ({idx + 1}/{total} rows processed)"
            )

    if extracted == 0:
        raise RuntimeError("no librispeech clips matched duration constraints")

    marker_payload = {
        "dataset_id": LIBRISPEECH_DATASET_ID,
        "config": LIBRISPEECH_CONFIG,
        "split": LIBRISPEECH_SPLIT,
        "clip_count": extracted,
        "skipped_rows": skipped,
        "snapshot_path": str(snapshot_path),
    }
    _ready_marker_path().write_text(
        json.dumps(marker_payload, indent=2),
        encoding="utf-8",
    )
    _CLIP_INDEX_CACHE.pop(audio_dir, None)
    logger.info(f"librispeech dataset extracted to {audio_dir} ({extracted} clips)")
    return audio_dir


def _list_clip_paths(audio_dir: Path) -> list[Path]:
    cached = _CLIP_INDEX_CACHE.get(audio_dir)
    if cached is not None:
        return [audio_dir / entry["file_name"] for entry in cached]

    paths = sorted(audio_dir.glob("*.wav"))
    _CLIP_INDEX_CACHE[audio_dir] = [{"file_name": path.name} for path in paths]
    return paths


def pick_random_audio_clips(
    count: int,
    *,
    audio_dir: Path | None = None,
) -> list[dict[str, bytes | str]]:
    directory = audio_dir or librispeech_audio_dir()
    if not directory.is_dir():
        raise ValueError(
            f"librispeech audio directory not found: {directory}. "
            "Run ensure_librispeech_dataset() during validator setup."
        )

    paths = _list_clip_paths(directory)
    if len(paths) < count:
        raise ValueError(
            f"need {count} librispeech clips, found {len(paths)} in {directory}"
        )

    selected = random.sample(paths, count)
    clips: list[dict[str, bytes | str]] = []
    for path in selected:
        clip_id = path.stem
        clips.append(
            {
                "id": clip_id,
                "audio_bytes": path.read_bytes(),
            }
        )
    return clips
