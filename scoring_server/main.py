from pathlib import Path
from typing import Optional

import base64
import tempfile

import requests  # optional if you later call other services; ok to keep
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from utils.media import download_video_from_url
from utils.audio_check import verify_video_audio_matches_text
from score.score import (
    MinerEvalInput,
    MinerEvalScores,
    evaluate_miner,
)
import os
from dotenv import load_dotenv
load_dotenv()

from loguru import logger

SCORING_SERVER_PORT = int(os.getenv("SCORING_SERVER_PORT", 8100))
RUN_MODE = os.getenv("RUN_MODE", "prod")

app = FastAPI(title="TalkHead Scoring Server")


# ---------- Request/Response models ---------- #

class ScoreRequest(BaseModel):
    # Required evaluation fields
    text: str = Field(..., description="The text to score.")
    language: str = Field("en-US", description="The language to score.")
    latency_sec: float = Field(..., description="The latency of the video in seconds.")

    video_url: str = Field(..., description="The URL of the video to score.")

    # NEW: image (from challenge)
    image_url: Optional[str] = Field(None, description="The URL of the reference face.")
    image_base64: Optional[str] = Field(None, description="The base64 encoded reference face.")
    voice_profile: Optional[str] = Field(None, description="Optional voice profile identifier.")

class ScoreResponse(BaseModel):
    ok: bool
    error_code: Optional[str]
    error_message: Optional[str]

    composite: float
    S_syncnet: float
    S_arcface: float
    S_quality: float
    # S_head_jerk: float
    # S_blink: float
    # S_flow: float
    # S_lpips: float
    latency_ratio: Optional[float]

    reason: str  # human-readable summary string


@app.get("/")
def health() -> dict[str, str]:
    return {"message": "server is alive and ready to score!"}


def _save_image_from_base64(b64_str: str) -> Path:
    data = base64.b64decode(b64_str)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _download_image(image_url: str) -> Path:
    resp = requests.get(image_url, stream=True, timeout=30)
    resp.raise_for_status()

    suffix = Path(image_url).suffix or ".png"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    for chunk in resp.iter_content(chunk_size=8192):
        if chunk:
            tmp.write(chunk)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:

    # 1. Basic validation
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    image_path: Path | None = None
    if req.image_base64:
        image_path = _save_image_from_base64(req.image_base64)
    elif req.image_url:
        try:
            image_path = _download_image(req.image_url)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid image_url: {e}")

    try:
        video_path = download_video_from_url(req.video_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid video_url: {e}")
    logger.info(f"🔵 Score request: {req.text} {req.language} {req.latency_sec} {req.video_url}  {req.voice_profile} {image_path} {video_path}")

    # 2. Build eval input for core scorer
    eval_input = MinerEvalInput(
        text=text,
        language=req.language,
        latency_sec=req.latency_sec,
        video_path=video_path,
        image_path=image_path,
        voice_profile=req.voice_profile,
    )

    # 3. Run evaluation
    if req.voice_profile:
        ok_audio, audio_reason = verify_video_audio_matches_text(
            text=text,
            voice_profile=req.voice_profile,
            video_path=video_path,
        )
        if not ok_audio:
            return ScoreResponse(
                ok=False,
                error_code="INCORRECT_AUDIO",
                error_message=audio_reason,
                composite=0.0,
                S_syncnet=0.0,
                S_arcface=0.0,
                S_quality=0.0,
                # S_head_jerk=0.0,
                # S_blink=0.0,
                # S_flow=0.0,
                # S_lpips=0.0,
                latency_ratio=None,
                reason="incorrect audio",
            )

    try:
        scores: MinerEvalScores = evaluate_miner(eval_input)
    except Exception as e:
        # make sure we don't crash the server
        return ScoreResponse(
            ok=False,
            error_code="EVAL_FAILED",
            error_message=str(e),
            composite=0.0,
            S_syncnet=0.0,
            S_arcface=0.0,
            S_quality=0.0,
            # S_head_jerk=0.0,
            # S_blink=0.0,
            # S_flow=0.0,
            # S_lpips=0.0,
            latency_ratio=None,
            reason="exception_during_evaluation",
        )

    return ScoreResponse(
        ok=True,
        error_code=None,
        error_message=None,
        composite=scores.composite,
        S_syncnet=scores.S_syncnet,
        S_arcface=scores.S_arcface,
        S_quality=scores.S_quality,
        # S_head_jerk=scores.S_head_jerk,
        # S_blink=scores.S_blink,
        # S_flow=scores.S_flow,
        # S_lpips=scores.S_lpips,
        latency_ratio=scores.latency_ratio,
        reason=scores.reason,
    )

if __name__ == "__main__":
    # run uvicorn main:app --reload
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=SCORING_SERVER_PORT, reload=(RUN_MODE == "dev"))