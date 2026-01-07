# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 TalkHead AI

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import numpy as np
import bittensor as bt

from talkhead.protocol import TalkHeadSynapse
from talkhead.validator.reward import get_rewards, apply_blended_rank, compute_latency_scores
from talkhead.utils.uids import get_available_uids
from talkhead.constants import TALKHEAD_SERVER_HOST, TESTNET_TALKHEAD_SERVER_HOST, DENDRITE_TIMEOUT
import requests
import base64
import wandb
from typing import List, Dict
from talkhead.utils.logging import maybe_reset_wandb

async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    
    miner_uids = get_available_uids(self)
    bt.logging.info(f"Available miners list length: {len(miner_uids)}")

    # randomly shuffle the available miners list.
    np.random.shuffle(miner_uids)
    
    total_uids = []
    total_composites = []
    total_detailed_metrics = []
    total_latency_ratios = []
    for i in range(0, max(len(miner_uids), self.config.neuron.sample_size), self.config.neuron.sample_size):
        selected_miner_uids = miner_uids[i:min(i + self.config.neuron.sample_size, len(miner_uids))]
        bt.logging.info(f"Selected miner uids: {selected_miner_uids}")
        
        host = TALKHEAD_SERVER_HOST if self.subtensor.network == "finney" else TESTNET_TALKHEAD_SERVER_HOST
        # Fetch the challenge from the talkhead server
        headers = self.build_signed_headers(f"/challenge")
        try:
            response = requests.get(f"{host}/challenge", headers=headers, timeout=10)
        except Exception as e:
            bt.logging.error(f"Failed to fetch challenge: {e}")
            continue
        challenge = response.json()

        bt.logging.info(f"🏁 Fetched a challenge => text: \'{challenge['text']}\' | voice_profile: \'{challenge['voice_profile']}\'")
        bt.logging.info(f"Querying {len(selected_miner_uids)} miners; {selected_miner_uids}")

        synapse = TalkHeadSynapse(image_base64=challenge["image_base64"], text=challenge["text"], voice_profile=challenge["voice_profile"])

        # The dendrite client queries the network.
        responses = await self.dendrite(
            # Send the query to selected miner axons in the network.
            axons=[self.metagraph.axons[uid] for uid in selected_miner_uids],
            # Construct a dummy query. This simply contains a single integer.
            synapse=synapse,
            # All responses have the deserialize function called on them before returning.
            # You are encouraged to define your own deserialization function.
            deserialize=True,
            timeout=DENDRITE_TIMEOUT,
        )

        # Log the results for monitoring purposes.
        bt.logging.info(f"🔵 Received responses: {responses}")

        valid_uids = []
        valid_responses = []

        for uid, (video_url, dendrite_process_time) in zip(selected_miner_uids, responses):
            if video_url is None or dendrite_process_time is None:
                bt.logging.debug(f"Invalid response: video_url: {video_url} | dendrite_process_time: {dendrite_process_time}")
                continue
            valid_uids.append(uid)
            valid_responses.append((video_url, dendrite_process_time))

        composites, detailed_metrics, latency_ratios = get_rewards(self, step=self.step, synapse=synapse, responses=valid_responses)

        for uid, composite, detailed_metric, latency_ratio in zip(valid_uids, composites, detailed_metrics, latency_ratios):
            bt.logging.info(f"🟣 Composite score for uid {uid}: {composite}")
            if composite > 0.0:
                total_uids.append(uid)
                total_composites.append(composite)
                total_detailed_metrics.append(detailed_metric)
                total_latency_ratios.append(latency_ratio)

    # Apply latency scores globally across all collected miners before ranking.
    total_latency_scores = compute_latency_scores(total_latency_ratios)
    final_rewards = [composite * latency_score for composite, latency_score in zip(total_composites, total_latency_scores)]
    bt.logging.debug(f"miner uids: {total_uids}")
    bt.logging.debug(f"total composites scores: {total_composites}")
    bt.logging.debug(f"Latency scores: {total_latency_scores}")
    bt.logging.debug(f"Final rewards: {final_rewards}")

    for idx, metrics in enumerate(total_detailed_metrics):
        metrics["composite_no_latency"] = float(total_composites[idx]) if idx < len(total_composites) else 0.0
        metrics["latency_ratio"] = total_latency_ratios[idx] if idx < len(total_latency_ratios) else None
        metrics["latency_score"] = float(total_latency_scores[idx]) if idx < len(total_latency_scores) else 0.0

    # Apply the blended ranking and quality threshold (always enabled).
    bt.logging.debug("Applying blended ranking and quality threshold to post-penalty rewards.")
    is_100_percent_burn = False
    applied_rewards, uids, detailed_metrics, is_100_percent_burn = apply_blended_rank(
        final_rewards,
        total_detailed_metrics,
        total_uids,
        top_miner_cap=self.top_miner_cap,
        decay_rate=self.decay_rate,
        blend_factor=self.blend_factor,
        burn_uid=self.burn_uid,
    )
    bt.logging.debug(f"Applied blended ranking uids: {uids}")
    bt.logging.debug(f"Applied blended ranking rewards: {applied_rewards}")

    if not is_100_percent_burn:
        do_wandb_logging(self, total_uids, total_composites, total_latency_scores, applied_rewards, detailed_metrics)

    # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
    self.update_scores(applied_rewards, uids)

    # prevent W&B logs from becoming massive
    maybe_reset_wandb(self)


def do_wandb_logging(
        self, 
        total_uids: List[int],
        total_composites: List[float],
        total_latency_scores: List[float],
        applied_rewards: List[float],
        detailed_metrics: List[Dict],
    ):
    if self.config.wandb.off:
        return

    uid_to_hotkey = {uid: self.metagraph.hotkeys[uid] for uid in total_uids}
    for uid, composite, latency_score, applied_reward, detailed_metric in zip(total_uids, total_composites, total_latency_scores, applied_rewards, detailed_metrics):
        wandb.log(
            {
                f"miner_{uid}_{uid_to_hotkey[uid]}_composite": composite,
                f"miner_{uid}_{uid_to_hotkey[uid]}_latency_score": latency_score,
                f"miner_{uid}_{uid_to_hotkey[uid]}_reward": applied_reward,
                f"miner_{uid}_{uid_to_hotkey[uid]}_syncnet": detailed_metric.get("S_syncnet", 0.0),
                f"miner_{uid}_{uid_to_hotkey[uid]}_arcface": detailed_metric.get("S_arcface", 0.0),
                f"miner_{uid}_{uid_to_hotkey[uid]}_quality": detailed_metric.get("S_quality", 0.0),
                f"miner_{uid}_{uid_to_hotkey[uid]}_reason": detailed_metric.get("reason", ""),
            },
        )