from pathlib import Path
import time

from score.score import evaluate_miner, MinerEvalInput
from utils.media import probe_duration
from score.metrics import (
    metric_syncnet,
    metric_arcface_identity,
    metric_quality,
    metric_head_jerk,
    metric_blink_rate,
    metric_raft_flow,
    metric_lpips,
)

cases = [
    # 0, # full test
    # 1, # syncnet only
    # 2, # arcface only
    3, # quality only
    # 4, # head jerk only
    # 5, # blink rate only
    # 6, # flow only 
    # 7, # lpips only
]

if __name__ == "__main__":
    for case in cases:
        if case == 0:
            start_time = time.time()
            scores = evaluate_miner(
                MinerEvalInput(
                    text="TalkHead Subnet is a Bittensor subnet focused on generating high-quality talking head avatars.",
                    language="en-US",
                    video_path=Path("../test_data/talker.mp4"),
                    image_path=Path("../test_data/talker.jpg"),
                    voice_profile="en_GB-alan-low",
                )
            )
            print(scores)
            end_time = time.time()
            print(f"Time taken: {end_time - start_time} seconds")
        if case == 1:  # syncnet
            start_time = time.time()
            S_syncnet, _ = metric_syncnet(Path("../test_data/talker.mp4"))
            print(f"S_syncnet: {S_syncnet} took {time.time() - start_time} seconds")
        if case == 2:  # arcface
            start_time = time.time()
            # S_arcface, _ = metric_arcface_identity(Path("../test_data/talker.jpg"), Path("../test_data/talker.mp4"))
            S_arcface, _ = metric_arcface_identity(Path("/tmp/tmpiq4pl66c.png"), Path("/tmp/tmp6la9efrc.mp4"))
            print(f"S_arcface: {S_arcface} took {time.time() - start_time} seconds")
        if case == 3:  # quality
            start_time = time.time()
            S_quality, _ = metric_quality(Path("../test_data/175212_you-could-have-died-after-peace-had-been_876aa703.mp4"))
            # S_quality, _ = metric_quality(Path("../test_data/063826_what-do-you-want-done-with-them-tomorrow_8b8b6558.mp4"))
            print(f"S_quality: {S_quality} took {time.time() - start_time} seconds")
        if case == 4:  # head jerk
            start_time = time.time()
            S_head_jerk, _ = metric_head_jerk(Path("../test_data/talker.mp4"))
            print(f"S_head_jerk: {S_head_jerk} took {time.time() - start_time} seconds")
        if case == 5:  # blink
            start_time = time.time()
            S_blink_rate, _ = metric_blink_rate(Path("../test_data/talker.mp4"))
            print(f"S_blink_rate: {S_blink_rate} took {time.time() - start_time} seconds")
        if case == 6:  # flow
            start_time = time.time()
            S_flow, _ = metric_raft_flow(Path("../test_data/talker.mp4"))
            print(f"S_flow: {S_flow} took {time.time() - start_time} seconds")
        if case == 7:  # lpips
            start_time = time.time()
            S_lpips, _ = metric_lpips(Path("../test_data/talker.mp4"))
            print(f"S_lpips: {S_lpips} took {time.time() - start_time} seconds")