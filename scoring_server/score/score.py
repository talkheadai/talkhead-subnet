from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from score.metric_weights import metric_weights
from score.metrics import (
    MetricResult,
    metric_syncnet,
    metric_arcface_identity,
    metric_quality,
    metric_head_jerk,
    metric_blink_rate,
    metric_raft_flow,
    metric_lpips,
)
from utils.media import probe_duration

@dataclass
class MinerEvalInput:
    text: str
    language: str
    video_path: Path
    voice_profile: Optional[str] = None
    image_path: Path | None = None


@dataclass
class MinerEvalScores:
    composite: float
    reason: str

    S_syncnet: float
    S_arcface: float
    S_quality: float
    # S_head_jerk: float
    S_blink: float
    # S_flow: float
    # S_lpips: float
    video_duration: float


def _normalize_weights(results: dict[str, MetricResult]) -> dict[str, float]:
    active_weights = {
        name: metric_weights[name]
        for name, res in results.items()
        if res.available and name in metric_weights
    }
    total = sum(active_weights.values())
    if total <= 0:
        return {}
    return {name: weight / total for name, weight in active_weights.items()}


def evaluate_miner(e: MinerEvalInput) -> MinerEvalScores:
    video_duration = probe_duration(e.video_path)
    if video_duration is None or video_duration <= 0.0:
        return MinerEvalScores(
            composite=0.0,
            reason="Failed to probe video duration",
            S_syncnet=0.0,
            S_arcface=0.0,
            S_quality=0.0,
            S_blink=0.0,
            video_duration=0.0,
        )
    blink_res, _ = metric_blink_rate(e.video_path)
    if blink_res.score == 0.0:
        return MinerEvalScores(
            composite=0.0,
            reason=blink_res.detail,
            S_syncnet=0.0,
            S_arcface=0.0,
            S_quality=0.0,
            S_blink=0.0,
            video_duration=video_duration,
        )
    quality_res, _ = metric_quality(e.video_path)
    if quality_res.score == 0.0:
        return MinerEvalScores(
            composite=0.0,
            reason=quality_res.detail,
            S_syncnet=0.0,
            S_arcface=0.0,
            S_quality=0.0,
            S_blink=0.0,
            video_duration=video_duration,
        )
    sync_res, _ = metric_syncnet(e.video_path)
    arc_res, _ = metric_arcface_identity(e.image_path, e.video_path)
    # head_res, _ = metric_head_jerk(e.video_path)
    # flow_res, _ = metric_raft_flow(e.video_path)
    # lpips_res, _ = metric_lpips(e.video_path)

    results = {
        "syncnet": sync_res,
        "arcface": arc_res,
        "quality": quality_res,
        # "head_jerk": head_res,
        "blink": blink_res,
        # "flow": flow_res,
        # "lpips": lpips_res,
    }
    applied_weights = _normalize_weights(results)
    composite = sum(results[name].score * applied_weights.get(name, 0.0) for name in results)

    reason_parts = []
    for name, res in results.items():
        raw_str = res.raw if res.raw is not None else "-"
        reason_parts.append(f"{name}={res.score:.2f} ({raw_str}; {res.detail})")
    if video_duration is not None:
        reason_parts.append(f"video_duration={video_duration:.2f}")
    else:
        reason_parts.append("video_duration=unavailable")
    reason = "; ".join(reason_parts)

    return MinerEvalScores(
        composite=float(composite),
        reason=reason,
        S_syncnet=sync_res.score,
        S_arcface=arc_res.score,
        S_quality=quality_res.score,
        # S_head_jerk=head_res.score,
        S_blink=blink_res.score,
        # S_flow=flow_res.score,
        # S_lpips=lpips_res.score,
        video_duration=video_duration,
    )
