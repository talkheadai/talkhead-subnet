# TalkHead Subnet

---

- [Overview](#overview)
- [How it works](#how-it-works)
- [How to Run](#how-to-run)
- [Executor & Scoring](#executor-and-scoring)

---

## Overview

TalkHead is a subnet where miners advertise Dockerized talking-head models on their axons, and validators evaluate them in a secure GPU executor to rank performance and set weights.

- Miners serve an immutable Docker image digest (`repo@sha256:...`) on their Bittensor axon via the `ImageRef` synapse.
- Validators discover registrations by querying miner axons directly through the metagraph.
- Model evaluation is performed externally by an executor service.
- Validators forward collected digests to the executor, read scores, and set on-chain weights.

## How it works

End-to-end pipeline:

1. Miner serves `ImageRef` on its axon (image digest pinned with `@sha256:`)
2. Validator queries serving miner axons for `ImageRef` responses
3. Validator -> Executor (`/update`) with collected digests
4. Executor evaluates submitted images
5. Executor provides scores (`/metrics`)
6. Validator sets weights on chain

## How to Run

### Requirements

- Python 3.11+
- A registered Bittensor wallet + hotkey
- Access to the executor API (validators only)
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

**Validator environment** (base + validator dependencies such as `wandb` and `numpy`):

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
- `EXECUTOR_API_URL` — executor base URL (validators only), e.g. `http://localhost:9000`

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

### Run Validator

The validator continuously:

1. Queries serving miner axons for `ImageRef` digests and forwards them to the executor
2. Reads executor metrics and sets on-chain weights

Start it with:

```bash
source .venv-validator/bin/activate
export EXECUTOR_API_URL=http://localhost:9000
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

## Executor and Scoring

Github repo => [talkheadai/talkhead-executor](https://github.com/talkheadai/talkhead-executor)

The executor handles model evaluation:

- Pulls miner Docker images.
- Runs images in a sandboxed environment.
- Sends challenge inputs and captures outputs.
- Measures performance and updates a score table.

Execution mechanics:

- File-based IPC using `/input` and `/output`.
- Warmup runs followed by scoring runs.
- No blockchain interaction.

Evaluation is standardized across miners:

- The executor evaluates all miners on the same challenge set.
- Runs include warmup and scoring phases.
- Lower latency produces a better score.

Round eligibility and carryover policy:

- Submissions that fail on Docker pull are excluded from the next evaluation round.
- Each round carries forward only:
  - The top 5 submissions by `final_score`.
  - New submissions received for the next round.

Validator behavior is split into two loops:

1. **Submission update loop**
  - Query serving miners on the metagraph for `ImageRef` digests.
  - Send collected digests to the executor.
2. **Weight setting loop**
  - Fetch scores from the executor.
  - Compute weights from score results.
  - Set weights on chain.

Winner-take-all policy:

- Highest score wins.
- Winning miner receives weight (minus any burn allocation).
- Burn allocation is defined by `BURN_RATIO` in `talkhead/constant.py`.
