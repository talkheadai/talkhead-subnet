from __future__ import annotations

import os
import threading
import time

from loguru import logger

from executor.challenges.loader import ChallengeLoader
from executor.evaluation.docker_runner import (
    evaluate,
    load_challenges_from_dir,
    stop_all_running_containers,
)
from executor.models import Challenge
from executor.state import MinerState

PENALTY_SCORE = 0


class EvaluationLoop:
    def __init__(
        self,
        state: MinerState,
        *,
        challenge_loader: ChallengeLoader | None = None,
        idle_sleep_sec: float = 1.0,
    ) -> None:
        self._state = state
        self._challenge_loader = challenge_loader or ChallengeLoader()
        self._idle_sleep_sec = idle_sleep_sec
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, name="evaluation-loop", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        stop_all_running_containers()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if not self._state.has_miners():
                time.sleep(self._idle_sleep_sec)
                continue

            try:
                challenges = self._fetch_challenges()
            except Exception as exc:
                logger.exception(f"failed to fetch challenges: {exc}")
                time.sleep(self._idle_sleep_sec)
                continue

            while not self._stop_event.is_set():
                pending = self._state.get_pending_miners()
                if not pending:
                    if self._state.commit_round_if_complete():
                        logger.info("round completed")
                    break

                for miner in pending:
                    logger.info(
                        f"evaluation start: hotkey={miner.hotkey} image_ref={miner.image_ref}"
                    )
                    metrics_payload: dict | None = None
                    score = PENALTY_SCORE
                    try:
                        score, metrics_payload = evaluate(miner.image_ref, challenges)
                    except Exception as exc:
                        logger.exception(
                            f"evaluation failed: hotkey={miner.hotkey} "
                            f"image_ref={miner.image_ref} error={exc}"
                        )
                        metrics_payload = {
                            "quality_score": float(PENALTY_SCORE),
                            "final_score": float(PENALTY_SCORE),
                            "challenge_metrics": [],
                            "error": str(exc),
                        }

                    metrics_payload = {
                        **metrics_payload,
                        "image_ref": miner.image_ref,
                        "updated_at": time.time(),
                    }
                    updated = self._state.set_evaluation_result(
                        hotkey=miner.hotkey,
                        image_ref=miner.image_ref,
                        metrics=metrics_payload,
                    )
                    if updated:
                        logger.info(
                            f"evaluation end: hotkey={miner.hotkey} "
                            f"image_ref={miner.image_ref} staged_final_score={score}"
                        )
                    else:
                        logger.info(
                            f"evaluation discarded due to updated submission: "
                            f"hotkey={miner.hotkey} image_ref={miner.image_ref}"
                        )

    def _fetch_challenges(self) -> list[Challenge]:
        challenges_dir = os.getenv("CHALLENGES_DIR", "").strip()
        if challenges_dir:
            challenges = load_challenges_from_dir(challenges_dir)
            logger.info(f"loaded {len(challenges)} challenges from {challenges_dir}")
            return challenges
        challenges = self._challenge_loader.load()
        logger.info(f"loaded {len(challenges)} challenges from APIs")
        return challenges
