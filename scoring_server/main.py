from pathlib import Path
from typing import Optional

import base64
import time

import requests  # optional if you later call other services; ok to keep
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from utils.media import download_video_from_url
from score.score import (
    MinerEvalInput,
    MinerEvalScores,
    evaluate_miner,
)
import os
from dotenv import load_dotenv
load_dotenv()

SCORING_SERVER_PORT = int(os.getenv("SCORING_SERVER_PORT", 8100))
RUN_MODE = os.getenv("RUN_MODE", "prod")

app = FastAPI(title="TalkHead Scoring Server")


# ---------- Request/Response models ---------- #

class ScoreRequest(BaseModel):
    # Required evaluation fields
    text: str = Field(..., description="The text to score.")
    language: str = Field("en-US", description="The language to score.")
    latency_sec: float = Field(..., description="The latency of the video in seconds.")
    duration_sec: Optional[float] = Field(None, description="The duration of the video in seconds.")

    video_url: str = Field(..., description="The URL of the video to score.")

    # NEW: reference face (from challenge)
    ref_face_url: Optional[str] = Field(None, description="The URL of the reference face.")
    ref_face_base64: Optional[str] = Field(None, description="The base64 encoded reference face.")

class ScoreResponse(BaseModel):
    ok: bool
    error_code: Optional[str]
    error_message: Optional[str]

    composite: float
    S_text: float
    S_duration: float
    S_latency: float
    S_sync: float
    S_face: float
    S_quality: float

    reason: str  # human-readable summary string


@app.get("/")
def health() -> dict[str, str]:
    return {"message": "server is alive and ready to score!"}

@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    # 1. Basic validation
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    try:
        video_path = download_video_from_url(req.video_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid video_url: {e}")

    # 2. Build eval input for core scorer
    eval_input = MinerEvalInput(
        text=text,
        language=req.language,
        latency_sec=req.latency_sec,
        video_path=video_path,
        target_duration_sec=req.duration_sec
    )

    # 3. Run evaluation
    try:
        scores: MinerEvalScores = evaluate_miner(eval_input)
    except Exception as e:
        # make sure we don't crash the server
        return ScoreResponse(
            ok=False,
            error_code="EVAL_FAILED",
            error_message=str(e),
            composite=0.0,
            S_text=0.0,
            S_duration=0.0,
            S_latency=0.0,
            S_sync=0.0,
            S_face=0.0,
            S_quality=0.0,
            reason="exception_during_evaluation",
        )

    return ScoreResponse(
        ok=True,
        error_code=None,
        error_message=None,
        composite=scores.composite,
        S_text=scores.S_text,
        S_duration=scores.S_duration,
        S_latency=scores.S_latency,
        S_sync=scores.S_sync,
        S_face=scores.S_face,
        S_quality=scores.S_quality,
        reason=scores.reason,
    )

if __name__ == "__main__":
    # run uvicorn main:app --reload
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=SCORING_SERVER_PORT, reload=(RUN_MODE == "dev"))