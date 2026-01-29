from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2

from score.syncnet import compute_syncnet_score
from score.faceid import _get_face_app, _load_image
from score.quality import score_quality
from utils.media import _sample_frames


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


@dataclass
class MetricResult:
    score: float
    raw: float | dict[str, float] | None
    available: bool
    detail: str


def metric_syncnet(
    video_path: Path,
) -> Tuple[MetricResult, Optional[float]]:
    """
    Metric 1: SyncNet confidence (reference-free by default).
    """
    try:
        score = float(compute_syncnet_score(video_path))
        return MetricResult(
            score=score,
            raw=None,
            available=True,
            detail="SyncNet score",
        ), score
    except Exception as exc:  # noqa: BLE001
        return MetricResult(
            score=0.0,
            raw=None,
            available=False,
            detail=f"SyncNet not available: {exc}",
        ), None

def metric_arcface_identity(
    ref_face_path: Path,
    video_path: Path,
    target: float = 0.88,
) -> Tuple[MetricResult, Optional[float]]:
    """
    Metric 3: ArcFace identity preservation.
    Uses reference image vs first+last frame embeddings.
    """
    try:
        app = _get_face_app()
        ref_img = _load_image(ref_face_path)
    except Exception as exc:  # noqa: BLE001
        return MetricResult(
            score=0.0,
            raw=None,
            available=False,
            detail=f"insightface unavailable: {exc}",
        ), None

    def _get_main_face(img):
        faces = app.get(img)
        if not faces:
            h, w = img.shape[:2]
            pad = max(32, int(0.1 * max(h, w)))
            padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            faces = app.get(padded)
        if not faces:
            return None
        faces.sort(key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
        return faces[0]

    ref_face = _get_main_face(ref_img)
    if not ref_face:
        return MetricResult(
            score=0.0,
            raw=None,
            available=True,
            detail="no face detected in reference image",
        ), None
    ref_emb = ref_face.normed_embedding

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return MetricResult(
            score=0.0,
            raw=None,
            available=True,
            detail="unable to read video",
        ), None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_frames = [0, max(0, frame_count - 1)]
    embs = []

    for idx in target_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        face = _get_main_face(frame)
        if not face:
            continue
        embs.append(face.normed_embedding)
    cap.release()

    if not embs:
        return MetricResult(
            score=0.0,
            raw=None,
            available=True,
            detail="no faces detected in first/last frame",
        ), None

    sims = [float(np.dot(ref_emb, e)) for e in embs]
    sim_mean = float(np.mean(sims))
    score = _clamp(sim_mean / target)
    return MetricResult(
        score=score,
        raw=sim_mean,
        available=True,
        detail=f"target >= {target}",
    ), sim_mean


def metric_head_jerk(video_path: Path, target: float = 0.12) -> Tuple[MetricResult, Optional[float]]:
    """
    Metric 4: Head jerk / motion smoothness.
    If mediapipe is available we approximate jerk using nose trajectory.
    """
    try:
        import mediapipe as mp  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return MetricResult(
            score=0.0,
            raw=None,
            available=False,
            detail=f"mediapipe not available: {exc}",
        ), None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return MetricResult(
            score=0.0,
            raw=None,
            available=True,
            detail="unable to read video",
        ), None

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    dt = 1.0 / max(fps, 1.0)
    positions = []

    with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if not result.pose_landmarks:
                continue
            nose = result.pose_landmarks.landmark[0]  # nose landmark
            positions.append((nose.x, nose.y, nose.z))
    cap.release()

    if len(positions) < 5:
        return MetricResult(
            score=0.0,
            raw=None,
            available=True,
            detail="insufficient pose landmarks to measure jerk",
        ), None

    def _smooth(arr: np.ndarray, k: int = 5) -> np.ndarray:
        if arr.shape[0] < k:
            return arr
        pad = k // 2
        padded = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
        kernel = np.ones(k) / k
        return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis=0, arr=padded)

    pos_arr = _smooth(np.array(positions), k=5)
    vel = np.gradient(pos_arr, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)
    jerk_mag = np.linalg.norm(jerk, axis=1)
    jerk_mean = float(np.mean(np.abs(jerk_mag)))
    # Cap extreme noise to keep score interpretable
    jerk_mean_capped = float(np.clip(jerk_mean, 0.0, 5.0))

    score = _clamp(1.0 - max(0.0, jerk_mean_capped - target) / (5.0 - target))
    return MetricResult(
        score=score,
        raw=jerk_mean_capped,
        available=True,
        detail=f"target <= {target} rad/s^3",
    ), jerk_mean_capped


def metric_blink_rate(video_path: Path, target_low: float = 10.0, target_high: float = 60.0) -> Tuple[MetricResult, Optional[float]]:
    """
    Metric 5: Blink rate naturalness using MediaPipe FaceMesh eye aspect ratio.
    - Target: 10-60 blinks/min
    - Computation: OpenFace AU45 or EMOCA detector on video frames.
    - Why it works for SadTalker: SadTalker includes blink simulation; ensures human-like frequency.
    - Anti-Gaming Aspect: Zero blinks → instant 0 (anti-static image gaming).
    """
    try:
        import mediapipe as mp  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return MetricResult(
            score=0.0,
            raw=None,
            available=False,
            detail=f"mediapipe not available: {exc}",
        ), None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return MetricResult(
            score=0.0,
            raw=None,
            available=True,
            detail="unable to read video",
        ), None

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_fps = 25.0
    stride = max(1, int(round(fps / target_fps)))
    frame_time = stride / max(fps, 1.0)

    left_eye_idx = [33, 160, 158, 133, 153, 144]
    right_eye_idx = [362, 385, 387, 263, 373, 380]
    blink_threshold = 0.3
    reopen_threshold = blink_threshold + 0.05
    drop_ratio = 0.92
    reopen_ratio = 0.95
    baseline_window = 30
    baseline_alpha = 0.1
    min_close_sec = 0.04
    min_close_frames = max(1, int(round(min_close_sec / max(frame_time, 1e-6))))

    def _eye_aspect_ratio(landmarks, idxs, w: int, h: int) -> Optional[float]:
        pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in idxs], dtype=np.float32)
        c = np.linalg.norm(pts[0] - pts[3])
        if c <= 1e-6:
            return None
        a = np.linalg.norm(pts[1] - pts[5])
        b = np.linalg.norm(pts[2] - pts[4])
        return float((a + b) / (2.0 * c))

    blink_count = 0
    closed_frames = 0
    in_blink = False
    prev_ear: Optional[float] = None
    min_ear_in_blink: Optional[float] = None
    baseline_samples: list[float] = []
    baseline_ear: Optional[float] = None
    total_frames_read = 0
    samples_processed = 0
    frames_with_face = 0
    max_samples = 600

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while samples_processed < max_samples:
            ret, frame = cap.read()
            if not ret:
                break
            if total_frames_read % stride != 0:
                total_frames_read += 1
                continue
            total_frames_read += 1
            samples_processed += 1
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            if not result.multi_face_landmarks:
                continue
            frames_with_face += 1
            landmarks = result.multi_face_landmarks[0].landmark
            left_ear = _eye_aspect_ratio(landmarks, left_eye_idx, w, h)
            right_ear = _eye_aspect_ratio(landmarks, right_eye_idx, w, h)

            if left_ear is None or right_ear is None:
                continue
            ear = (left_ear + right_ear) / 2.0
            if baseline_ear is None:
                baseline_samples.append(ear)
                baseline_ear = float(np.median(baseline_samples))
            elif not in_blink:
                baseline_ear = (1.0 - baseline_alpha) * baseline_ear + baseline_alpha * ear

            close_thresh = baseline_ear * drop_ratio if baseline_ear is not None else blink_threshold
            open_thresh = baseline_ear * reopen_ratio if baseline_ear is not None else reopen_threshold
            
            if not in_blink:
                if ear < close_thresh:
                    closed_frames += 1
                    if closed_frames >= min_close_frames:
                        in_blink = True
                        min_ear_in_blink = ear
                else:
                    closed_frames = 0
            else:
                if min_ear_in_blink is not None:
                    min_ear_in_blink = min(min_ear_in_blink, ear)
                if ear >= open_thresh:
                    blink_count += 1
                    in_blink = False
                    closed_frames = 0
                    min_ear_in_blink = None

            prev_ear = ear

    cap.release()

    duration_sec = frame_count / max(fps, 1.0) if frame_count > 0 else total_frames_read * (1.0 / max(fps, 1.0))
    if duration_sec <= 0 or frames_with_face < 5:
        return MetricResult(
            score=0.0,
            raw=None,
            available=True,
            detail="insufficient face landmarks to estimate blink rate",
        ), None

    if blink_count == 0:
        return MetricResult(
            score=0.0,
            raw=None,
            available=True,
            detail="no blinks detected",
        ), None
    blink_rate = float(blink_count / duration_sec * 60.0)
    if blink_rate <= 0.0:
        score = 0.0
    elif blink_rate < target_low:
        score = _clamp(blink_rate / target_low)
    elif blink_rate <= target_high:
        score = 1.0
    else:
        score = _clamp(1.0 - (blink_rate - target_high) / target_high)

    return MetricResult(
        score=score,
        raw=blink_rate,
        available=True,
        detail=f"target {target_low}-{target_high} blinks/min",
    ), blink_rate


def metric_raft_flow(video_path: Path, target: float = 2.8) -> Tuple[MetricResult, Optional[float]]:
    """
    Metric 6: Temporal consistency via optical flow.
    If RAFT is unavailable, fall back to OpenCV Farneback flow and compute
    average backward warping error (px).
    """
    try:
        import cv2  # ensure present
    except Exception as exc:  # noqa: BLE001
        return MetricResult(
            score=0.0,
            raw=None,
            available=False,
            detail=f"opencv not available: {exc}",
        ), None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return MetricResult(
            score=0.0,
            raw=None,
            available=True,
            detail="unable to read video",
        ), None

    prev_gray = None
    errs = []
    max_pairs = 64
    pair_count = 0

    while pair_count < max_pairs:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray
            continue

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        h, w = prev_gray.shape
        grid_y, grid_x = np.mgrid[0:h, 0:w]
        map_x = (grid_x + flow[:, :, 0]).astype(np.float32)
        map_y = (grid_y + flow[:, :, 1]).astype(np.float32)
        warped = cv2.remap(gray, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        err = np.mean(np.abs(prev_gray.astype(np.float32) - warped.astype(np.float32)))
        errs.append(err)

        prev_gray = gray
        pair_count += 1

    cap.release()

    if not errs:
        return MetricResult(
            score=0.0,
            raw=None,
            available=True,
            detail="no frame pairs to compute flow",
        ), None

    avg_err = float(np.mean(errs))
    # Map error: <= target -> 1, >= 2*target -> 0
    if avg_err <= target:
        score = 1.0
    else:
        score = _clamp((2 * target - avg_err) / target)

    return MetricResult(
        score=score,
        raw=avg_err,
        available=True,
        detail=f"avg warping error px (target <= {target})",
    ), avg_err

def metric_lpips(video_path: Path, target: float = 0.095) -> Tuple[MetricResult, Optional[float]]:
    """
    Metric 8: Self-perceptual quality via LPIPS after 4x Real-ESRGAN upscale.
    Steps:
      - sample frames
      - upscale with Real-ESRGAN x4
      - compare LPIPS between bicubic-upscaled original and Real-ESRGAN output
    """
    frames = _sample_frames(video_path, max_frames=8)
    if not frames:
        return MetricResult(
            score=0.0,
            raw=None,
            available=True,
            detail="no frames for LPIPS",
        ), 0.0

    try:
        import torch
        import lpips  # type: ignore
        from realesrgan.utils import RealESRGANer  # type: ignore
        from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
    except Exception as exc:  # noqa: BLE001
        # Fallback: simple sharpness proxy when LPIPS/ESRGAN unavailable
        laps = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laps.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        if not laps:
            return MetricResult(
                score=0.0,
                raw=None,
                available=True,
                detail=f"LPIPS fallback failed: {exc}",
            ), 0.0
        sharp = float(np.mean(laps))
        if sharp >= 300:
            score = 1.0
        elif sharp <= 50:
            score = 0.0
        else:
            score = _clamp((sharp - 50) / (300 - 50))
        return MetricResult(
            score=score,
            raw=sharp,
            available=True,
            detail="LPIPS fallback (no lpips/realesrgan): sharpness proxy",
        ), sharp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esr = None
    weight_url = "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth"
    last_exc: Exception | None = None
    try:
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        esr = RealESRGANer(
            scale=4,
            model_path=weight_url,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False,
            device=device,
        )
    except Exception as exc:  # noqa: BLE001
        last_exc = exc
        esr = None
        
    if esr is None:
        return MetricResult(
            score=0.0,
            raw=None,
            available=False,
            detail=f"failed to init Real-ESRGANer: {last_exc}",
        ), None

    lpips_net = lpips.LPIPS(net="alex").to(device)

    dists = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            sr, _ = esr.enhance(rgb, outscale=4)
        except Exception:
            sr = rgb
        sr = np.clip(np.array(sr), 0, 255).astype(np.uint8)
        h, w, _ = rgb.shape
        base = cv2.resize(rgb, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)

        def to_tensor(img: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0)
            t = t.permute(2, 0, 1).unsqueeze(0)
            return t.to(device)

        t_base = to_tensor(base)
        t_sr = to_tensor(sr)
        with torch.no_grad():
            dist = float(lpips_net(t_base, t_sr).cpu().item())
        dists.append(dist)

    if not dists:
        return MetricResult(
            score=0.0,
            raw=None,
            available=True,
            detail="LPIPS computed no distances",
        ), 0.0

    avg_lpips = float(np.mean(dists))

    if avg_lpips <= target:
        score = 1.0
    else:
        score = _clamp((0.2 - avg_lpips) / (0.2 - target))

    return MetricResult(
        score=score,
        raw=avg_lpips,
        available=True,
        detail=f"LPIPS vs ESRGAN x4 (target <= {target})",
    ), avg_lpips


def metric_quality(video_path: Path) -> Tuple[MetricResult, Optional[float]]:
    """
    Wrapper around legacy quality score (sharpness/brightness/motion/audio).
    """
    try:
        val = float(score_quality(video_path))
    except Exception as exc:  # noqa: BLE001
        return MetricResult(
            score=0.0,
            raw=None,
            available=False,
            detail=f"quality score failed: {exc}",
        ), None

    return MetricResult(
        score=_clamp(val),
        raw=val,
        available=True,
        detail="legacy quality score",
    ), val
