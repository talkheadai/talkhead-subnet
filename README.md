<div align="center">
# **TalkHead Subnet** <!-- omit in toc -->
<video  src="https://github.com/user-attachments/assets/f8dd6afc-d2d2-4a0b-a4d9-49e51dd4aeb0" type="video/mp4"> </video>
</div>

# TalkHead Subnet

## Overview

TalkHead is a subnet for evaluating talking-head generation models through Docker-based execution.

- Miners submit Docker image digests that identify model runtime containers.
- Model evaluation is performed externally by an executor service.
- Validators coordinate submission updates, scoring intake, and on-chain weight setting.

## Architecture

The system is composed of four roles:

1. **Miner**
   - Builds and publishes a Docker image for talking-head generation.
   - Submits the image digest to the subnet coordination layer.

2. **Validator**
   - Collects miner submissions and forwards them for evaluation.
   - Reads scores and sets on-chain weights.

3. **Executor**
   - Pulls submitted images, runs challenge evaluations, and computes scores.
   - Maintains evaluation outputs used by validators.

4. **Subnet API**
   - Stores submissions and challenge metadata.

## System Flow

End-to-end pipeline:

1. Miner -> Subnet API (`/submit`)
2. Validator -> Subnet API (`/submissions`)
3. Validator -> Executor (`/update`)
4. Executor evaluates submitted images
5. Executor provides scores
6. Validator sets weights on chain

## Miner

Miner responsibilities are intentionally narrow:

- Build a Docker image using the `talkhead-miner-image` repository.
- Push the image to a container registry.
- Submit the image digest through the subnet API.

Miner does **not**:

- Run GPU inference inside the subnet node runtime.
- Expose serving endpoints to validators.

The miner role is submission-only.

## Validator

Validator behavior is split into two loops.

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

## Executor

The executor handles model evaluation:

- Pulls miner Docker images.
- Runs images in a sandboxed environment.
- Sends challenge inputs and captures outputs.
- Measures performance and updates a score table.

Execution mechanics:

- File-based IPC using `/input` and `/output`.
- Warmup runs followed by scoring runs.
- No blockchain interaction.

## Subnet API

The subnet API is a thin coordination service:

- Stores miner submissions.
- Provides challenge data.

## Evaluation Model

Evaluation is standardized across miners:

- The executor evaluates all miners on the same challenge set.
- Runs include warmup and scoring phases.
- Lower latency produces a better score.

## Repositories

- `talkhead-subnet` (this repository)
- `talkhead-miner-image`
- `executor` (separate repository)
- `subnet API` (separate service)

## Notes
