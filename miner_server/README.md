# TalkHead Miner Server

Minimal FastAPI wrapper that turns the local [SadTalker](https://github.com/sadtalker/SadTalker)
pipeline into an HTTP service for the TalkHead subnet.

## Quick start

```bash
cd miner_server
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
# launch the API
uvicorn miner_server.main:app --host 0.0.0.0 --port 9000 --reload
```

The editable install keeps your virtualenv pointed at the local source tree so
code changes are reflected immediately.

## SadTalker assets

This repository already vendors the `miner_server/SadTalker` tree. To refresh
weights or fetch the latest upstream assets run:

```bash
cd miner_server/SadTalker
bash scripts/download_models.sh
```

Make sure `ffmpeg`/`ffprobe` binaries are available on your `PATH`; SadTalker
relies on them for audio/video processing.

## Optional extras

- `pip install -e .[dev]` installs pytest and helpers for local testing.

## Sample request

```bash
curl -X POST http://localhost:9000/generate \
  -H "Content-Type: application/json" \
  -d '{
        "script": "TalkHead miners turn audio and portraits into talking heads.",
        "image_url": "https://i.postimg.cc/DyH90PGg/hat.jpg"
      }' \
  | jq -r .video_base64 | base64 -d > output.mp4
```

The endpoint responds with base64-encoded MP4 bytes (`video_base64`) and basic
error fields (`ok`, `error_code`, `error_message`).

## Troubleshooting

- `SADTALKER_ROOT` is inferred automatically (`miner_server/SadTalker`). If you
  relocate it, update the constant in `sadtalker_backend.py`.
- GPU acceleration depends on the upstream SadTalker install and PyTorch build.
  Install the appropriate CUDA wheels before calling the API.
- The API logs errors rather than crashing; check the HTTP response for details
  if generation fails.

