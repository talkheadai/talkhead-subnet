# TalkHead Scoring Server

This folder can now be installed as an independent Python package so you can
spin up a local scoring API without pulling in the rest of the monorepo.

## How to setup

### Install g++

```bash
apt-get update -y
apt-get install g++ -y
apt install ffmpeg -y
```

### Setup python virtual env and python packages

```bash
cd scoring_server
python -m venv .venv
source .venv/bin/activate
pip install -e .
```
### Optional extras

- `pip install -e .[dev]` adds pytest and other utilities.
- `pip install -e .[gpu]` swaps the default CPU `onnxruntime` wheel for the
  CUDA-enabled build (plus the matching CUDA runtime) so InsightFace can run
  inference on GPU. If you switch to the GPU build later, uninstall the CPU
  wheel first:

  ```bash
  pip uninstall -y onnxruntime
  pip install -e .[gpu]
  ```
- `pip install -e .[metrics]` pulls in optional deps for jerk/blink/LPIPS
  scaffolding (mediapipe, lpips, realesrgan). You still need to provide model
  weights where required.

### Download syncnet weight files

```bash
wget https://huggingface.co/lithiumice/syncnet/resolve/main/sfd_face.pth -P score/syncnet
wget https://huggingface.co/lithiumice/syncnet/resolve/main/syncnet_v2.model -P score/syncnet
```

#### Wav2Lip for LSE-D/C
- Submodule (preferred):  
  ```bash
  git submodule update --init --recursive Wav2Lip
  ```
  (or clone: `git clone https://github.com/Rudrabha/Wav2Lip.git Wav2Lip`)
- If the submodule entry is missing (fresh fork), add it once from repo root:
  ```bash
  git submodule add https://github.com/Rudrabha/Wav2Lip.git scoring_server/Wav2Lip
  ```
- Install its deps: `pip install -r Wav2Lip/requirements.txt`
- Download checkpoints into the repo:
  ```bash
  # Install gdown once if you don't have it
  pip install gdown

  # Wav2Lip model (non-GAN)
  gdown --id 15G3U08c8xsCkOqQxE38Z2XXDnPcOptNk -O Wav2Lip/checkpoints/wav2lip.pth

  
  wget --no-check-certificate 'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth' -O Wav2Lip/checkpoints/s3fd.pth
  ```
- Nothing else is needed; the scorer auto-adds this path to `sys.path` if present.

#### LPIPS / Real-ESRGAN weights
- LPIPS installs via pip; Real-ESRGAN x4 weights auto-download to `~/.cache/realesrgan/RealESRGAN_x4plus.pth`.  
  Override with `REALESRGAN_X4_URL` if you host the weight elsewhere.

## Running the API

Once dependencies are installed you can launch the FastAPI app with uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8100
```
or just use
```bash
python main.py
```

### Using pm2
```bash
pm2 start --name scoring_server main.py
```
### Setup .env file
```
cp .env.example .env
```
`Edit .env file to customize the PORT and RUN_MODE and restart your api server`

### API contract (current)

- Request body:  
  - `text` (str), `language` (str, default `en-US`), `latency_sec` (float), `video_url` (str)
  - optional `ref_face_url` or `ref_face_base64`
  - optional `voice_profile` (Piper voice id; enables audio verification)
- Response body:  
  - `composite`
  - per-metric `S_` scores: `S_syncnet`, `S_lse`, `S_arcface`, `S_head_jerk`, `S_blink`, `S_flow`, `S_lpips`
  - `latency_ratio` (latency_sec / video_duration_sec)
  - `reason` (human-readable summary)

### Audio verification (voice_profile)

- If `voice_profile` is provided, the scorer synthesizes reference audio with
  Piper using that profile and the provided `text`, then compares it to the
  video's extracted audio.
- If similarity is below threshold, the request returns `ok=false` with
  `error_code=INCORRECT_AUDIO`, `composite=0`, all `S_*=0`, and `reason`
  `"incorrect audio"`.
- Piper voices are searched under `PIPER_VOICE_DIR` (defaults to
  `scoring_server/piper-voices`). Ensure the matching `<voice_profile>.onnx`
  exists there.

The server exposes a single `/score` endpoint that accepts the payload defined
in `main.ScoreRequest` and returns the composite scoring response.

### Metric breakdown (validator-side, reference-free)

| # | Metric | Weight | Target | Notes |
|---|--------|--------|--------|-------|
|1|SyncNet confidence|30%|≥ 0.93|Uses SyncNet-v2; optionally supply validator TTS audio.|
|2|Lip sync error (LSE-D/C)|25%|LSE-D ≤ 1.8 px, LSE-C ≥ 0.85|Requires Wav2Lip distance; marked unavailable until wired.|
|3|ArcFace identity|20%|Cosine ≥ 0.88|Reference image vs first+last frame embeddings.|
|4|Head jerk / smoothness|10%|≤ 0.12 rad/s³|Needs MediaPipe Pose; unavailable if missing.|
|5|Blink rate naturalness|5%|10–22 blinks/min|Requires blink detector (OpenFace/EMOCA).|
|6|Temporal consistency (RAFT)|5%|Warp error ≤ 2.8 px|Requires RAFT optical flow.|
|7|Latency bonus|3%|≤ 8s, full at 5s|Linear bonus from 8s→5s.|
|8|Self-perceptual quality (LPIPS)|2%|LPIPS ≤ 0.095|After 4× Real-ESRGAN; optional deps.|

If a metric’s dependency is missing, it is marked “unavailable” and its weight
is re-normalized across the metrics that did run. The response returns the
composite score, per-metric `S_` scores, and a human-readable `reason`.

### Notes

- The package automatically ships the SyncNet weight files stored under
  `score/syncnet/`.
- ffmpeg/ffprobe binaries must be available on your `PATH` for media probing
  and audio extraction to work.


