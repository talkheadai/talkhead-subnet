import base64
import io
from typing import Literal, Optional
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, Field
from PIL import Image

from generate import generate_talking_head
import os
from dotenv import load_dotenv

load_dotenv()
MINER_SERVER_PORT = int(os.getenv("MINER_SERVER_PORT", 9000))
RUN_MODE = os.getenv("RUN_MODE", "prod")

app = FastAPI(title="TalkHead Miner HTTP API")


# ---------- Request / Response models ---------- #

class GenerateRequest(BaseModel):
    text: str = Field(..., description="The text to generate a talking head for.")

    # One of these must be provided:
    image_url: Optional[HttpUrl] = Field(None, description="The URL of the image to use for the talking head.")
    image_base64: Optional[str] = Field(None, description="The base64 encoded image to use for the talking head.")

    # Optional hints:
    language: str = Field("en-US", description="The language to use for the talking head.")
    voice_profile: str = Field("neutral", description="The voice profile to use for the talking head.")
    duration_sec: Optional[float] = Field(8.0, description="The duration of the talking head in seconds.")


class GenerateResponse(BaseModel):
    ok: bool = Field(..., description="Whether the generation was successful.")
    error_code: Optional[str] = Field(None, description="The error code if the generation was not successful.")
    error_message: Optional[str] = Field(None, description="The error message if the generation was not successful.")

    # Base64-encoded video bytes for now (you can switch to streaming/url later)
    video_base64: Optional[str] = Field(None, description="The base64 encoded video bytes for the talking head.")
    mime_type: Literal["video/mp4"] = Field("video/mp4", description="The mime type of the video.")


# ---------- Helper functions ---------- #

def _load_image_bytes_from_url(url: str) -> bytes:
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {e}")
    return resp.content


def _load_image_bytes_from_base64(data: str) -> bytes:
    try:
        return base64.b64decode(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")


def _validate_image(image_bytes: bytes) -> bytes:
    """
    Ensure it's a valid image and maybe do a simple sanity check.
    Returns the original bytes if valid.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()  # just validation, not decoding fully
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    return image_bytes


# ---------- Routes ---------- #
@app.get("/")
def health() -> dict[str, str]:
    return {"message": "server is alive and ready to generate!"}

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    # 1. Validate text
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    # 2. Load image bytes
    if req.image_url:
        image_bytes = _load_image_bytes_from_url(str(req.image_url))
    elif req.image_base64:
        image_bytes = _load_image_bytes_from_base64(req.image_base64)
    else:
        raise HTTPException(
            status_code=400,
            detail="Either image_url or image_base64 must be provided.",
        )

    # 3. Validate image
    image_bytes = _validate_image(image_bytes)

    # 4. Call core generator (currently stub)
    try:
        video_bytes = generate_talking_head(
            image_bytes=image_bytes,
            text=text,
            voice_profile=req.voice_profile,
        )
    except FileNotFoundError as e:
        # clear, explicit error if sample video missing
        return GenerateResponse(
            ok=False,
            error_code="GEN_STUB_MISSING_SAMPLE_VIDEO",
            error_message=str(e),
            video_base64=None,
        )
    except Exception as e:
        return GenerateResponse(
            ok=False,
            error_code="GENERATION_FAILED",
            error_message=str(e),
            video_base64=None,
        )

    # 5. Encode video as base64
    video_b64 = base64.b64encode(video_bytes).decode("ascii")

    return GenerateResponse(
        ok=True,
        error_code=None,
        error_message=None,
        video_base64=video_b64,
    )


if __name__ == "__main__":
    # run uvicorn main:app --reload
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=MINER_SERVER_PORT, reload=(RUN_MODE == "dev"))