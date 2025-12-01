# TalkHead Miner Server

Minimal FastAPI wrapper that turns the local [SadTalker](https://github.com/sadtalker/SadTalker)
pipeline into an HTTP service for the TalkHead subnet.

## How to setup

### Set up SadTalker

This repository already vendors the `miner_server/SadTalker` tree. To refresh weights or fetch the latest upstream assets run:

```bash
cd miner_server
git submodule update --init --recursive SadTalker
cd SadTalker
bash scripts/download_models.sh
cd ..
```

### Install ffmpeg
```bash
apt update -y
apt install ffmpeg -y
```

Make sure `ffmpeg`/`ffprobe` binaries are available on your `PATH`; SadTalker
relies on them for audio/video processing.

### Set up python virtual env and install python packages

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Optional extras

- `pip install -e .[dev]` installs pytest and helpers for local testing.


## How to run the api

```bash
uvicorn miner_server.main:app --host 0.0.0.0 --port 9000 --reload
```
or just use
```bash
python main.py
```

### Using pm2
```bash
pm2 start --name miner_server main.py
```
### Setup .env file
```
cp .env.example .env
```
`Edit .env file to customize the PORT and RUN_MODE and restart your api server`



## Troubleshooting

- `SADTALKER_ROOT` is inferred automatically (`miner_server/SadTalker`). If you
  relocate it, update the constant in `sadtalker_backend.py`.
- GPU acceleration depends on the upstream SadTalker install and PyTorch build.
  Install the appropriate CUDA wheels before calling the API.
- The API logs errors rather than crashing; check the HTTP response for details
  if generation fails.


 - If you face with the following error
`Could not load library libcudnn_cnn_infer.so.8. Error: libcuda.so: cannot open shared object file: No such file or directory`, simply run the following commands
```bash
ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
ldconfig
```