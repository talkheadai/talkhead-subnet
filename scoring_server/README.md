# TalkHead Scoring Server

This folder can now be installed as an independent Python package so you can
spin up a local scoring API without pulling in the rest of the monorepo.

## Quick start

```bash
cd scoring_server
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

The editable install makes your virtualenv point directly at the local source,
so any code changes are picked up immediately without reinstalling.

## Optional extras

- `pip install -e .[dev]` adds pytest and other utilities.
- `pip install -e .[gpu]` swaps the default CPU `onnxruntime` wheel for the
  CUDA-enabled build (plus the matching CUDA runtime) so InsightFace can run
  inference on GPU. If you switch to the GPU build later, uninstall the CPU
  wheel first:

  ```bash
  pip uninstall -y onnxruntime
  pip install -e .[gpu]
  ```

## Download weight files

```bash
wget https://huggingface.co/lithiumice/syncnet/blob/main/sfd_face.pth -P syncnet
wget https://huggingface.co/lithiumice/syncnet/blob/main/syncnet_v2.model -P syncnet
```

## Running the API

Once dependencies are installed you can launch the FastAPI app with uvicorn:

```bash
uvicorn scoring_server.main:app --host 0.0.0.0 --port 8100 --reload
```

The server exposes a single `/score` endpoint that accepts the payload defined
in `scoring_server.main.ScoreRequest` and returns the composite scoring
response.

## Notes

- The package automatically ships the SyncNet weight files stored under
  `scoring_server/syncnet/`.
- ffmpeg/ffprobe binaries must be available on your `PATH` for media probing
  and audio extraction to work.


