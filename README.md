# TalkHead Subnet

---

- [Overview](#overview)
- [How it works](#how-it-works)
- [How to Run](#how-to-run)
- [Executor & Scoring](#executor-and-scoring)

---

## Overview

TalkHead is a subnet where miners submit Dockerized talking-head models, and validators evaluate them in a secure GPU executor to rank performance and set weights.

- Miners submit Docker image digests that identify model runtime containers.
- Model evaluation is performed externally by an executor service.
- Validators coordinate submission updates, scoring intake, and on-chain weight setting.

## How it works

End-to-end pipeline:

1. Miner -> Subnet API (`/submit`)
2. Validator -> Subnet API (`/submissions`)
3. Validator -> Executor (`/update`)
4. Executor evaluates submitted images
5. Executor provides scores
6. Validator sets weights on chain

## How to Run

### Requirements

- Python 3.11+
- A registered Bittensor wallet + hotkey
- Access to the subnet API and executor endpoints
- A published miner image digest in `repo@sha256:...` format

### Setup

Install the project and create a local environment file:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Set the required values in `.env`:

- `WALLET_NAME` and `HOTKEY_NAME` for the registered wallet/hotkey
- `NETWORK` and `NETUID` for the target subnet
- `SUBNET_API_URL` for the coordination API
- `EXECUTOR_API_URL` for the executor API used by the validator
- `IMAGE_REF` for the miner's published Docker image digest in `repo@sha256:...` format

### Run Miner

The miner is submission-only. It sends the configured Docker image digest to the subnet API.

You can use the [talkheadai/talkhead-miner-image](https://github.com/talkheadai/talkhead-miner-image) repository as a base Docker image/template for your miner container.

```bash
python -m neurons.miner
```

You can also override the image ref from the CLI:

```bash
python -m neurons.miner --image-ref your-registry/your-image@sha256:...
```

> [!NOTE]
> Submissions from the same hotkey must be at least two days apart; submitting again sooner returns HTTP **429** (Too Many Requests) from the subnet API.

### Run Validator

The validator continuously:

1. Pulls miner submissions from the subnet API and forwards them to the executor
2. Reads executor metrics and sets on-chain weights

Start it with:

```bash
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

Validator behavior is split into two loops:

1. **Submission update loop**
  - Fetch submissions from the subnet API.
  - Send updated miner image digests to the executor.
2. **Weight setting loop**
  - Fetch scores from the executor.
  - Compute weights from score results.
  - Set weights on chain.

Winner-take-all policy:

- Highest score wins.
- Winning miner receives weight.
- All other miners receive zero.

