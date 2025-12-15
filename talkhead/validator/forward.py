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
from talkhead.validator.reward import get_rewards, apply_blended_rank
from talkhead.utils.uids import get_available_uids
from talkhead.constants import TALKHEAD_SERVER_ENDPOINT, DENDRITE_TIMEOUT
import requests
import base64

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
    
    total_rewards = []
    total_detailed_metrics = []
    for i in range(0, max(len(miner_uids), 4), 4):
        selected_miner_uids = miner_uids[i:min(i + 4, len(miner_uids))]
        bt.logging.info(f"Selected miner uids: {selected_miner_uids}")

        # Fetch the challenge from the talkhead server
        response = requests.get(TALKHEAD_SERVER_ENDPOINT + "/challenge")
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

        rewards, detailed_metrics = get_rewards(self, step=self.step, synapse=synapse, responses=responses)
        bt.logging.info(f"🟣 Scored responses: {rewards}")

        total_rewards.extend(rewards)
        total_detailed_metrics.extend(detailed_metrics)

    # Get burn configuration from config with defaults
    burn_fraction = getattr(self.config.neuron, 'burn_fraction', 0)
    burn_uid = 59  # Hardcoded: burn UID is always 59 and never configurable
    keep_fraction = 1.0 - burn_fraction
    
    # Apply the blended ranking and quality threshold (always enabled).
    bt.logging.debug("Applying blended ranking and quality threshold to post-penalty rewards.")
    is_100_percent_burn = False
    applied_rewards, selected_miner_uids, detailed_metrics, is_100_percent_burn = apply_blended_rank(
        total_rewards,
        total_detailed_metrics,
        miner_uids,
        top_miner_cap=self.config.neuron.top_miner_cap,
        decay_rate=self.config.neuron.decay_rate,
        blend_factor=self.config.neuron.blend_factor,
        burn_uid=burn_uid,
    )
    bt.logging.debug(f"Applied blended ranking and quality threshold to post-penalty rewards. {applied_rewards}")
    bt.logging.debug(f"Applied blended ranking and quality threshold to post-penalty rewards. {miner_uids}")

    # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
    self.update_scores(applied_rewards, miner_uids)
    
