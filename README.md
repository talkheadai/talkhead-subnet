<div align="center">

# **TalkHead Subnet** <!-- omit in toc -->
[Discord](https://discord.com/channels/799672011265015819/1450185343205904565) • [Network](https://talkhead.ai/) • [Research](https://docs.google.com/document/d/1vOLAdBdgtxUn0n8izYjUQ4bS8DX6KhvUjvw9Zk3e28s/edit?usp=sharing)
</div>

---
- [About](#about)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Validator Setup & Run](#validator-setup--run)
- [Configure Validator](#configure-validator)
- [Miner Setup & Run](#miner-setup--run)
- [Configure Miner](#configure-miner)
- [Rewards and scoring](#rewards-and-scoring)
- [License](#license)

---
## About

TalkHead is a Bittensor subnet that incentivizes miners to generate high-quality, lip-synced talking-head video clips from text plus a reference image (and optional voice profile). Validators evaluate realism and identity preservation, then apply a latency bonus after quality scoring before setting on-chain weights.

The repo ships both the blockchain-facing neurons (miner/validator) and the off-chain services used to generate and score videos.

---
## How it works

- **Forward**: Validators query miners with a TalkHead Synapse, which carries `image_base64`, `text`, and `voice_profile`.
- **Generate**: Miners forward the TalkHead Synapse to their own talking-head video generation API to render a clip, upload it to Cloudflare R2, and return a public video URL.
- **Score**: Validators send the video URL to the scoring server, which computes quality metrics (sync, identity, motion, etc.) and returns a composite quality score.
- **Latency bonus**: Validators apply a latency bonus separately using the dendrite processing time versus video duration; latency is not part of the composite.
- **Reward**: Scores are blended with rank-based decay and an optional emission burn before updating validator weights on-chain.
---
## Requirements

- Python 3.10+ and a Linux host with CUDA-capable GPU. See `min_compute.yml` for suggested CPU/GPU/RAM/network specs.
- `ffmpeg`/`ffprobe` and `g++` installed system-wide.
- Optional: Cloudflare R2 credentials if you want miners to upload clips (`CLOUDFLARE_R2_*`), Hugging Face CLI for Piper voices, CUDA-enabled onnxruntime for GPU scoring.

> [!NOTE]
> The entire project was tested on an RTX 4090 with no errors observed.
---
## Validator Setup & Run

You can run your own scoring server using the [scoring_server installation guide](./scoring_server/README.md).

### Set up a validator neuron

From repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Run a validator neuron
```bash
python neurons/validator.py --wallet.name [your_wallet_name] --wallet.hotkey [your_wallet_hotkey]
```

Using pm2
```bash
pm2 start neurons/validator.py -- --wallet.name [your_wallet_name] --wallet.hotkey [your_wallet_hotkey]
```

### Configure Validator

Key environment variables (override defaults as needed):

| Variable | Purpose | Default |
| --- | --- | --- |
| `SCORING_SERVER_ENDPOINT` | Scoring API endpoint (`/score`) | `http://localhost:8100/generate` |
| `DENDRITE_TIMEOUT` | Validator query timeout (seconds) | `120` |

Validator reward knobs (CLI flags, defaults shown):
- `--neuron.top_miner_cap 2` – how many miners are eligible for rank-based rewards.
- `--neuron.decay_rate 0.05` – exponential decay applied to rank.
- `--neuron.blend_factor 0.7` – blend between rank reward and raw score.

Validators will pull challenges, query miners, call the scoring API, then set weights each epoch.


## Miner Setup & Run

Run a miner server using the [miner_server installation guide](./miner_server/README.md).

### Set up a miner neuron

From repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Run a miner neuron
```bash
python neurons/miner.py --wallet.name [your_wallet_name] --wallet.hotkey [your_wallet_hotkey]
```

Using pm2
```bash
pm2 start neurons/miner.py -- --wallet.name [your_wallet_name] --wallet.hotkey [your_wallet_hotkey]
```

---
## Configure Miner

Key environment variables (override defaults as needed):

| Variable | Purpose | Default |
| --- | --- | --- |
| `MINER_SERVER_ENDPOINT` | generation API endpoint(`/generate`) | `http://localhost:9000/generate` |
| `CLOUDFLARE_R2_BUCKET` / `CLOUDFLARE_R2_ACCESS_KEY_ID` / `CLOUDFLARE_R2_SECRET_ACCESS_KEY` | R2 upload credentials (optional) | unset |
| `CLOUDFLARE_R2_PUBLIC_BASE_URL` / `CLOUDFLARE_R2_ENDPOINT` or `CLOUDFLARE_R2_ACCOUNT_ID` | Public URL + endpoint for R2 | unset |
---

## Rewards and scoring

- Composite quality score uses `scoring_server` metrics: SyncNet confidence, ArcFace identity, motion/quality checks (jerk/blink/flow/LPIPS when available). Latency is applied as a post-score bonus on the validator.
- `apply_blended_rank()` caps to the top `--neuron.top_miner_cap` miners, blends rank-based decay with raw scores, and optionally burns emissions to UID 0 when no miner qualifies.
- Rewards feed into the validator’s exponential moving average before setting weights on-chain.

---
## License

MIT License © 2023 Yuma Rao, © 2025 TalkHead AI. See [LICENSE](./LICENSE).

