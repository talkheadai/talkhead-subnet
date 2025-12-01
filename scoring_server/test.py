from score.score import evaluate_miner, MinerEvalInput
from score.score import score_sync, score_face, score_quality
from pathlib import Path

testing_cases = [
    0, # all mertrics
    # 1, # text only
    # 2, # lip sync only
    # 3, # face id only
    # 4, # quality only
]

if __name__ == "__main__":
    for case in testing_cases:
        if case == 0:
            scores = evaluate_miner(MinerEvalInput(
                miner_id="test",
                text="TalkHead Subnet is a Bittensor subnet focused on generating high-quality talking head avatars.",
                language="en-US",
                latency_ms=1000,
                video_path=Path("../test_data/talker.mp4"),
                target_duration_sec=8.0,
                ref_face_path=Path("../test_data/talker.jpg"),
            ))
            print(scores)
        if case == 2:
            print(score_sync(Path("../test_data/hat.mp4")))
        if case == 3:
            print(score_face(Path("../test_data/cute.jpg"), Path("../test_data/cute.mp4")))
        if case == 4:
            print(score_quality(Path("../test_data/hat.mp4")))