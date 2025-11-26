from pathlib import Path
from typing import Optional

import base64
import time

import requests  # optional if you later call other services; ok to keep
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from scoring_server.media_utils import save_video_base64_to_temp
from scoring_server.scoring import (
    MinerEvalInput,
    MinerEvalScores,
    evaluate_miner,
)


app = FastAPI(title="TalkHead Scoring Server")


# ---------- Request/Response models ---------- #

class ScoreRequest(BaseModel):
    # Required evaluation fields
    script: str
    language: str = "en-US"
    latency_ms: float
    target_duration_sec: float = 8.0

    video_base64: str

    # Optional metadata
    miner_id: Optional[str] = None
    challenge_id: Optional[str] = None

    # NEW: reference face (from challenge)
    ref_face_url: Optional[str] = None
    ref_face_base64: Optional[str] = None

class ScoreResponse(BaseModel):
    ok: bool
    error_code: Optional[str]
    error_message: Optional[str]

    miner_id: Optional[str]
    challenge_id: Optional[str]

    composite: float
    S_script: float
    S_duration: float
    S_latency: float
    S_sync: float
    S_face: float
    S_quality: float

    reason: str  # human-readable summary string

@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    # 1. Basic validation
    script = req.script.strip()
    if not script:
        raise HTTPException(status_code=400, detail="script must not be empty")

    try:
        video_path = save_video_base64_to_temp(req.video_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid video_base64: {e}")

    # 2. Build eval input for core scorer
    eval_input = MinerEvalInput(
        miner_id=req.miner_id or "unknown",
        script=script,
        language=req.language,
        latency_ms=req.latency_ms,
        video_path=video_path,
        target_duration_sec=req.target_duration_sec,
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
            miner_id=req.miner_id,
            challenge_id=req.challenge_id,
            composite=0.0,
            S_script=0.0,
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
        miner_id=scores.miner_id,
        challenge_id=req.challenge_id,
        composite=scores.composite,
        S_script=scores.S_script,
        S_duration=scores.S_duration,
        S_latency=scores.S_latency,
        S_sync=scores.S_sync,
        S_face=scores.S_face,
        S_quality=scores.S_quality,
        reason=scores.reason,
    )

if __name__ == "__main__":
    # test the API
    import json
    response = requests.post("http://localhost:8100/score", json={
        "script": "TalkHead Subnet is a Bittensor subnet focused on generating high-quality talking head avatars. This subnet incentivizes miners to produce realistic, synchronized talking head videos from audio input and reference images, enabling applications in virtual avatars, video generation, and AI-driven content creation.",
        # "image_url": "https://i.postimg.cc/1tj4SZbT/cute.jpg",
        "image_url": "https://i.postimg.cc/DyH90PGg/hat.jpg",
        "latency_ms": 1000,
        "target_duration_sec": 24.0,
        "video_base64": base64.b64encode(open("hat.mp4", "rb").read()).decode("ascii"),
        "miner_id": "test",
        "challenge_id": "test",
    })
    print(json.dumps(response.json(), indent=4))