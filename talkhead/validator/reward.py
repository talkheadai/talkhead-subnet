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
from typing import List, Tuple, Dict
import bittensor as bt
from talkhead.constants import SCORING_SERVER_ENDPOINT
import requests
from talkhead.protocol import TalkHeadSynapse

def reward(step: int, synapse: TalkHeadSynapse, video_url: str, dendrite_process_time: float) -> Tuple[float, Dict]:
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
        "image_base64": synapse.image_base64,
        "language": "en-US",
        "voice_profile": synapse.voice_profile,
        "latency_sec": float(dendrite_process_time),
    }

    try:
        scoring_response = requests.post(
            SCORING_SERVER_ENDPOINT,
            json=payload,
            timeout=60,
        )
        scoring_response.raise_for_status()
        result = scoring_response.json()
    except requests.RequestException as err:
        bt.logging.error(f"Failed to score miner response: {err}")
        return 0.0, {"reason": "Failed to score miner response"}
    except ValueError:
        bt.logging.error("Scoring server returned non-JSON response.")
        return 0.0, {"reason": "Scoring server returned non-JSON response"}

    composite_score = result.get("composite")
    bt.logging.debug(f"Scoring server response => composite_score: {composite_score} | reason: {result.get('reason', 'No reason provided')}")
    if composite_score is None:
        bt.logging.error(f"Scoring server response missing 'composite': {result}")
        return 0.0, {"reason": "Scoring server response missing 'composite'"}

    return float(composite_score), result

def get_rewards(
    self,
    step: int,
    synapse: TalkHeadSynapse,
    responses: List[TalkHeadSynapse],
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Returns an array of rewards for the given query and responses.

    Args:
    - step (int): The current validator step.
    - responses (List[TalkHeadSynapse]): A list of responses from the miner.

    Returns:
    - np.ndarray: An array of rewards for the given query and responses.
    """
    # Get all the reward results by iteratively calling your reward() function.

    rewards, detailed_metrics = zip(*[reward(step, synapse, video_url, dendrite_process_time) for video_url, dendrite_process_time in responses])
    return np.array(rewards), list(detailed_metrics)

def apply_blended_rank(rewards: List[float], detailed_metrics: List[Dict], uids: List[int], top_miner_cap: int, decay_rate: float, blend_factor: float, burn_uid: int) -> Tuple[np.ndarray, np.ndarray, List[Dict], bool]:
    """
    Applies a blended ranking model with a quality threshold.
    1. Ranks all miners based on their initial reward scores.
    2. Filters for miners within the top_miner_cap.
    4. Re-ranks the final qualified group and calculates an exponential decay reward.
    5. Blends the rank-based reward with the original reward score.
    6. Miners who do not qualify receive a reward of 0.
    
    Args:
        burn_uid: UID to receive burned emissions (always 59, hardcoded)
    """
    # Convert uids to numpy array for consistent return type
    uids = np.array(uids)
    
    # Get global ranks (0 = best) based on the post-penalty rewards
    rewards = np.array(rewards)

    global_ranks = (-rewards).argsort().argsort()

    # Stage 1: Identify miners within top cap
    within_cap_mask = global_ranks < top_miner_cap
    within_cap_indices = np.where(within_cap_mask)[0]
    final_rewards = np.zeros_like(rewards)
    # Update detailed metrics for logging and analysis (before burn check)
    for i, metrics in enumerate(detailed_metrics):
        metrics['ranking_info'] = {
            'initial_reward': float(rewards[i]), # This is now the post-penalty reward
            'global_rank': int(global_ranks[i]),
            'within_top_cap': bool(within_cap_mask[i]),
            'final_blended_reward': 0.0  # Will be updated if miner qualifies
        }
    if len(within_cap_indices) > 0:
        # Get original rewards for the qualified miners
        within_cap_original_rewards = rewards[within_cap_indices]
        # Re-rank ONLY among the qualified miners
        within_cap_ranks = (-within_cap_original_rewards).argsort().argsort()
        # Calculate exponential decay reward based on rank
        ranked_rewards_component = np.exp(-decay_rate * within_cap_ranks)
        # Blend the rank reward with the original reward
        blended_rewards = (blend_factor * ranked_rewards_component + 
                           (1 - blend_factor) * within_cap_original_rewards)
        # Place the calculated blended rewards into the final rewards array
        final_rewards[within_cap_indices] = blended_rewards
        # Update the final_blended_reward for qualified miners
        for idx, within_cap_idx in enumerate(within_cap_indices):
            detailed_metrics[within_cap_idx]['ranking_info']['final_blended_reward'] = float(blended_rewards[idx])
        bt.logging.info(f"Applied blended ranking: {within_cap_mask.sum()} of {len(rewards)} miners within top cap for rewards.")
        # Return False to indicate 100% burn did not occur (miners qualified)
        return final_rewards, uids, detailed_metrics, False
    else:
        # No miners qualified - burn all emissions
        bt.logging.warning("🔥 BURN EVENT: No miners within top cap after applying top cap")
        bt.logging.warning(f"All emissions will be burned to UID {burn_uid}")
        
        # Extend arrays to include burn UID
        extended_uids = np.append(uids[within_cap_indices], burn_uid)
        extended_rewards = np.append(final_rewards, 1.0)  # All zeros except burn UID gets 1.0
        
        # Update detailed metrics to include burn UID
        burn_metrics = {
            'uid': burn_uid,
            'is_burn': True,
            'burn_type': '100_percent',  # Indicates 100% burn due to no miners within top cap
            'ranking_info': {
                'initial_reward': 0.0,
                'global_rank': -1,  # Special rank for burn
                'within_top_cap': False,
                'final_blended_reward': 1.0  # Burn gets all rewards
            }
        }
        detailed_metrics.append(burn_metrics)
        
        # Return True to indicate 100% burn occurred (no miners qualified)
        return extended_rewards, extended_uids, detailed_metrics, True
