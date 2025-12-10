from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import insightface
import torch
from loguru import logger


@lru_cache(maxsize=1)
def _get_face_app():
    """
    Load InsightFace model once. Uses GPU if available, else CPU.
    """
    # name='buffalo_l' is a good default: detector + recognizer
    app = insightface.app.FaceAnalysis(name="buffalo_l")
    # ctx_id: 0 for GPU 0, -1 for CPU
    app.prepare(ctx_id=0 if torch.cuda.is_available() else -1,
                det_size=(640, 640))
    return app


def _load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to read image at {path}")
    logger.info(f"img.ndim: {img.ndim}")
    logger.info(f"img.shape: {img.shape}")
    # InsightFace expects 3-channel BGR input; some PNGs are BGRA or grayscale.
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _get_face_embedding(img: np.ndarray) -> np.ndarray | None:
    """
    Return a single 512-d embedding for the *main* face in the image,
    or None if no face detected.
    """
    app = _get_face_app()
    faces = app.get(img)
    if not faces:
        # Some close-cropped faces fail detection; pad to give detector context.
        h, w = img.shape[:2]
        pad = max(32, int(0.1 * max(h, w)))
        padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        faces = app.get(padded)
    if not faces:
        return None

    # pick largest face
    faces.sort(key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
    emb = faces[0].normed_embedding  # already L2-normalized
    return np.asarray(emb, dtype=np.float32)


def _sample_video_frames(video_path: Path, max_frames: int = 16) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        cap.release()
        return []

    step = max(1, frame_count // max_frames)
    frames = []
    idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if len(frames) >= max_frames:
            break
        idx += step

    cap.release()
    return frames


def _get_video_embeddings(video_path: Path, max_frames: int = 16) -> List[np.ndarray]:
    frames = _sample_video_frames(video_path, max_frames=max_frames)
    if not frames:
        return []

    app = _get_face_app()
    embs: List[np.ndarray] = []

    for frame in frames:
        faces = app.get(frame)
        if not faces:
            continue
        faces.sort(key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
        emb = np.asarray(faces[0].normed_embedding, dtype=np.float32)
        embs.append(emb)

    return embs


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def compute_face_identity_score(
    ref_face_path: Path,
    video_path: Path,
) -> float:
    """
    Face identity consistency in [0,1]:
      - ref consistency: similarity between ref face and video faces
      - stability: similarity between video faces across time
    """

    # 1) Reference embedding
    ref_img = _load_image(ref_face_path)
    ref_emb = _get_face_embedding(ref_img)
    if ref_emb is None:
        # can't find face in reference → we can't judge identity
        return 0.0

    # 2) Video embeddings
    video_embs = _get_video_embeddings(video_path, max_frames=16)
    if not video_embs:
        return 0.0

    embs = np.stack(video_embs, axis=0)  # (N, 512), already unit norm
    mean_emb = np.mean(embs, axis=0)
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)

    # 3) Identity consistency with reference
    sims_ref = [ _cosine(ref_emb, e) for e in embs ]
    sim_ref_mean = float(np.mean(sims_ref))  # typically in [-1,1]

    # 4) Stability: each frame vs mean embedding
    sims_stab = [ _cosine(mean_emb, e) for e in embs ]
    sim_stab_mean = float(np.mean(sims_stab))

    # 5) Map cosine [-1,1] -> [0,1]
    def norm_cos(x: float) -> float:
        return max(0.0, min((x + 1.0) / 2.0, 1.0))

    S_ref = norm_cos(sim_ref_mean)
    S_stab = norm_cos(sim_stab_mean)

    S_face = 0.6 * S_ref + 0.4 * S_stab
    return float(max(0.0, min(S_face, 1.0)))
