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
import numpy as np
from typing import List
import bittensor as bt
from talkhead.constants import SCORING_SERVER_ENDPOINT
import requests
from talkhead.protocol import TalkHeadSynapse


def reward(step: int, synapse: TalkHeadSynapse, video_url: str) -> float:
    """
    Reward the miner response to the challenge request. This method returns a reward
    value for the miner, which is used to update the miner's score.

    Args:
    - step (int): The current validator step.
    - synapse (TalkHeadSynapse): The synapse object containing the challenge request and miner response.

    Returns:
    - float: The reward value for the miner.
    """
    if not video_url:
        bt.logging.error("Received response without video; assigning zero reward.")
        return 0.0

    payload = {
        "text": synapse.text,
        "video_url": video_url,
        "ref_face_base64": synapse.image_base64,
        "language": "en-US",
        "voice_profile": synapse.voice_profile,
    }

    try:
        scoring_response = requests.post(
            SCORING_SERVER_ENDPOINT + "/score",
            json=payload,
            timeout=60,
        )
        scoring_response.raise_for_status()
        result = scoring_response.json()
    except requests.RequestException as err:
        bt.logging.error(f"Failed to score miner response: {err}")
        return 0.0
    except ValueError:
        bt.logging.error("Scoring server returned non-JSON response.")
        return 0.0

    composite_score = result.get("composite")
    bt.logging.info(f"Scoring server response: {result.get('reason', 'No reason provided')}")
    if composite_score is None:
        bt.logging.error(f"Scoring server response missing 'composite': {result}")
        return 0.0

    return float(composite_score)

def get_rewards(
    self,
    step: int,
    synapse: TalkHeadSynapse,
    responses: List[TalkHeadSynapse],
) -> np.ndarray:
    """
    Returns an array of rewards for the given query and responses.

    Args:
    - step (int): The current validator step.
    - responses (List[TalkHeadSynapse]): A list of responses from the miner.

    Returns:
    - np.ndarray: An array of rewards for the given query and responses.
    """
    # Get all the reward results by iteratively calling your reward() function.

    return np.array([reward(step, synapse, video_url) for video_url in responses])
