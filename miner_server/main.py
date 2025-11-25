import base64
import io
from typing import Literal, Optional
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from PIL import Image

from miner_server.generate import generate_talking_head


app = FastAPI(title="TalkHead Miner HTTP API")


# ---------- Request / Response models ---------- #

class GenerateRequest(BaseModel):
    script: str

    # One of these must be provided:
    image_url: Optional[HttpUrl] = None
    image_base64: Optional[str] = None

    # Optional hints:
    language: str = "en-US"
    voice_profile: str = "neutral"
    duration_sec: Optional[float] = 8.0


class GenerateResponse(BaseModel):
    ok: bool
    error_code: Optional[str]
    error_message: Optional[str]

    # Base64-encoded video bytes for now (you can switch to streaming/url later)
    video_base64: Optional[str]
    mime_type: Literal["video/mp4"] = "video/mp4"


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

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    # 1. Validate script
    script = req.script.strip()
    if not script:
        raise HTTPException(status_code=400, detail="Script must not be empty.")

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
            script=script,
            duration_sec=req.duration_sec,
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
    # test the API
    response = requests.post("http://localhost:9000/generate", json={
        "script": "Hello, world!",
        "image_url": "https://i.postimg.cc/1tj4SZbT/cute.jpg",
    })
    with open(f"cute.mp4", "wb") as f:
        f.write(base64.b64decode(response.json()["video_base64"]))
    print(f"ok: {response.json()['ok']}, error_code: {response.json()['error_code']}, error_message: {response.json()['error_message']}")