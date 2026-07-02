# TalkHead Subnet

---

- [Overview](#overview)
- [How it works](#how-it-works)
- [How to Run](#how-to-run)
- [Executor & Scoring](#executor-and-scoring)

---

## Overview

TalkHead is a subnet where miners advertise Dockerized talking-head models on their axons, and validators evaluate them locally to rank performance and set weights.

- Miners serve an immutable Docker image digest (`repo@sha256:...`) on their Bittensor axon via the `ImageRef` synapse.
- Validators discover registrations by querying miner axons directly through the metagraph.
- The executor runs **in-process** inside the validator: a background thread pulls miner Docker images, scores them, and stores results in local SQLite state.
- Validators read scores from that state and set on-chain weights.

## How it works

End-to-end pipeline:

1. Miner serves `ImageRef` on its axon (image digest pinned with `@sha256:`)
2. Validator queries serving miner axons for `ImageRef` responses
3. Validator upserts collected digests into in-process executor state
4. Executor background thread fetches challenges (Pexels faces + Freesound speech audio), runs Docker evaluation, and writes scores
5. Validator reads metrics from executor state
6. Validator sets weights on chain

## How to Run

### Requirements

- Python 3.11+
- Docker with GPU support (validators)
- A registered Bittensor wallet + hotkey
- Pexels and Freesound API keys (validators), or a local `CHALLENGES_DIR` for offline testing
- A published miner image digest in `repo@sha256:...` format

### Setup

Miner and validator use separate Python environments, both installed from the same `pyproject.toml` via optional dependency groups.

Create a local environment file:

```bash
cp .env.example .env
```

**Miner environment** (base dependencies only):

```bash
python -m venv .venv-miner
source .venv-miner/bin/activate
pip install -e ".[miner]"
```

**Validator environment** (base + executor/scoring dependencies):

```bash
python -m venv .venv-validator
source .venv-validator/bin/activate
pip install -e ".[validator]"
```

To install both apps in one environment:

```bash
pip install -e ".[miner,validator]"
```

Set the required values in `.env`:

- `IMAGE_REF` — miner's published Docker image digest in `repo@sha256:...` format
- `PEXELS_API_KEY` — portrait photos for evaluation challenges (validators)
- `FREESOUND_API_KEY` — speech/voice audio clips for evaluation challenges (validators)
- `STATE_FILE` — SQLite path for executor state (default: `./state.db`)

Wallet, network, and netuid are configured via standard Bittensor CLI flags (e.g. `--wallet.name`, `--wallet.hotkey`, `--subtensor.network`, `--netuid`).

### Run Miner

The miner runs an axon that responds to validator `ImageRef` queries with the configured Docker image digest. Update your model by changing `IMAGE_REF` (or `--image-ref`) and restarting the miner.

You can use the [talkheadai/talkhead-miner-image](https://github.com/talkheadai/talkhead-miner-image) repository as a base Docker image/template for your miner container.

```bash
source .venv-miner/bin/activate
python -m neurons.miner
```

You can also override the image ref from the CLI:

```bash
python -m neurons.miner --image-ref your-registry/your-image@sha256:...
```

> [!NOTE]
> Validators enforce a **4-day resubmission cooldown** per hotkey. If a miner advertises a new `image_ref` on its axon within 4 days of the last accepted update, the validator keeps the previous digest in executor state until the cooldown expires. Change `RESUBMIT_COOLDOWN_DAYS` in `talkhead/constant.py` to adjust this window.

### Run Validator

The validator runs the executor in-process and continuously:

1. Queries serving miner axons for `ImageRef` digests and upserts them into executor state
2. Reads executor metrics from local state and sets on-chain weights

A background evaluation thread handles Docker scoring while the main loop handles axon queries and weight setting.

```bash
source .venv-validator/bin/activate
export PEXELS_API_KEY=your-pexels-key
export FREESOUND_API_KEY=your-freesound-key
export WANDB_API_KEY=your-wandb-api-key
python -m neurons.validator
```

Common CLI overrides:

```bash
python -m neurons.validator \
  --wallet.name default \
  --wallet.hotkey default \
  --subtensor.network finney \
  --netuid 108
```

Optional tuning via environment variables:

- `MINER_QUERY_BATCH_SIZE` — axon queries per batch (default: `16`)
- `MINER_QUERY_TIMEOUT` — seconds to wait per axon query (default: `12`)
- `CHALLENGE_COUNT` — challenges per evaluation round (default: `7`)
- `CHALLENGES_DIR` — use local challenge fixtures instead of Pexels/Freesound APIs

## Executor and Scoring

The `executor/` package lives in this repo and is embedded in the validator process.

The executor handles model evaluation:

- Pulls miner Docker images.
- Runs images in a sandboxed environment.
- Sends challenge inputs and captures outputs.
- Measures performance and updates a score table in SQLite.

Execution mechanics:

- File-based IPC using `/input` and `/output`.
- Warmup runs followed by scoring runs.
- No blockchain interaction.

Challenge generation:

- Default: `ChallengeLoader` fetches portrait photos from **Pexels** and filters Freesound results for human speech (`tag:speech`, `tag:english`, `avg_rating >= 4`, blocked robot/alien/crowd tags). Set `FREESOUND_SOUND_IDS` for a fixed vetted clip pool.
- Offline/dev: set `CHALLENGES_DIR` to a directory of `face.png` + `audio.wav` subfolders.

Evaluation is standardized across miners:

- The executor evaluates all miners on the same challenge set per round.
- Runs include warmup and scoring phases.
- Lower latency produces a better score.

Round eligibility and carryover policy:

- Submissions that fail on Docker pull are excluded from the next evaluation round.
- A miner may change its advertised `image_ref` at most once every 4 days per hotkey; earlier changes are ignored by the validator.
- Each round carries forward only:
  - The top 5 submissions by `final_score`.
  - New submissions received for the next round.

Validator behavior is split into two loops:

1. **Submission update loop**
  - Query serving miners on the metagraph for `ImageRef` digests.
  - Upsert digests into in-process executor state (`updated_time` set on first sight or when a new digest is accepted).
  - Reject `image_ref` changes inside the 4-day cooldown (`RESUBMIT_COOLDOWN_DAYS` in `talkhead/constant.py`).
2. **Weight setting loop**
  - Read scores from executor state.
  - Compute weights from score results.
  - Set weights on chain.

Winner-take-all policy:

- Highest score wins.
- Winning miner receives weight (minus any burn allocation).
- Burn allocation is defined by `BURN_RATIO` in `talkhead/constant.py`.

> **Note:** `executor/app.py` (standalone HTTP server) is deprecated. Run `python -m neurons.validator` instead.
