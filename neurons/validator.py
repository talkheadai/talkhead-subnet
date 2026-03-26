from __future__ import annotations

import json
import logging
import math
import os
import time
from urllib import error, request

import bittensor as bt
import click
import numpy as np

from utils import signed_subnet_headers

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
INTERVAL_BLOCKS = 1
SUBNET_API_URL= os.getenv("SUBNET_API_URL", "https://subnet.talkhead.ai")

def _http_json(
    url: str,
    method: str,
    body: object | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> tuple[int, object]:
    payload = None
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
    req = request.Request(url, data=payload, method=method)
    req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)

    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw) if raw else None
    except error.HTTPError as http_err:
        raw = http_err.read().decode("utf-8")
        try:
            return http_err.code, json.loads(raw) if raw else None
        except json.JSONDecodeError:
            return http_err.code, raw
    except Exception as exc:  # noqa: BLE001
        return 0, str(exc)


class Validator:
    def __init__(self) -> None:
        self.netuid = int(os.getenv("NETUID", "108"))
        network = os.getenv("NETWORK", "finney")
        wallet_name = os.getenv("WALLET_NAME", "default")
        wallet_hotkey = os.getenv("HOTKEY_NAME", "default")
        self.executor_url = os.getenv("EXECUTOR_API_URL", "http://localhost:9000")
        self.wallet = bt.Wallet(name=wallet_name, hotkey=wallet_hotkey)
        self.subtensor = bt.Subtensor(network=network)
        self.metagraph = bt.Metagraph(netuid=self.netuid, network=network)
        self.burn_uid = self.metagraph.hotkeys.index(
            self.subtensor.subnet(netuid=self.netuid).owner_hotkey
        )
        if self.burn_uid < 0 or self.burn_uid >= len(self.metagraph.hotkeys):
            logger.warning(f"Burn UID out of range: {self.burn_uid}")
            self.burn_uid = 0

    @staticmethod
    def _pick_winner(scores: list[dict]) -> tuple[str, float] | None:
        valid: list[tuple[str, float]] = []
        for row in scores:
            if not isinstance(row, dict):
                continue
            hotkey = row.get("hotkey")
            score = row.get("score")
            if not isinstance(hotkey, str):
                continue
            try:
                value = float(score)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            valid.append((hotkey, value))

        if not valid:
            return None
        return max(valid, key=lambda item: item[1])

    def _sync_and_get_current_block(self) -> int:
        self.metagraph.sync(subtensor=self.subtensor)
        return self.subtensor.get_current_block()

    def _submission_update_step(self) -> None:
        logger.info("Fetching submissions")
        status, submissions = _http_json(
            f"{SUBNET_API_URL.rstrip('/')}/submissions",
            "GET",
            headers=signed_subnet_headers(self.wallet, "/submissions"),
        )
        if status < 200 or status >= 300 or not isinstance(submissions, list):
            logger.warning(
                f"Failed to fetch submissions (status={status}): {submissions}"
            )
            return
        if len(submissions) == 0:
            logger.info("No submissions received; skipping update")
            return

        logger.info("Updating executor")
        exec_status, exec_response = _http_json(
            f"{self.executor_url.rstrip('/')}/update",
            "POST",
            headers=signed_subnet_headers(self.wallet, "/update"),
            body=submissions,
        )
        if not (200 <= exec_status < 300):
            logger.warning(
                f"Executor update failed (status={exec_status}): {exec_response}"
            )

    def _weight_setting_step(self) -> None:
        logger.info("Fetching scores")
        score_status, scores = _http_json(
            f"{self.executor_url.rstrip('/')}/scores",
            "GET",
            headers=signed_subnet_headers(self.wallet, "/scores"),
        )
        if (
            score_status < 200
            or score_status >= 300
            or not isinstance(scores, list)
            or len(scores) == 0
        ):
            logger.info("No scores available; burning all")
            self._set_burn_only_weights()
            return

        winner = self._pick_winner(scores)
        if winner is None:
            logger.info("All scores invalid; skipping weight update")
            return
        winner_hotkey, _winner_score = winner

        logger.info("Fetching burn ratio")
        burn_ratio = 1.0
        burn_status, burn_response = _http_json(
            f"{SUBNET_API_URL.rstrip('/')}/burn_ratio",
            "GET",
            headers=signed_subnet_headers(self.wallet, "/burn_ratio"),
        )
        if (
            burn_status < 200
            or burn_status >= 300
            or not isinstance(burn_response, dict)
        ):
            logger.warning(
                f"Failed to fetch burn ratio (status={burn_status}): {burn_response}"
            )

        burn_ratio_value = burn_response.get("burn_ratio")
        try:
            burn_ratio = max(0.0, min(1.0, float(burn_ratio_value)))
        except (TypeError, ValueError):
            logger.warning(f"Invalid burn ratio payload: {burn_response}")

        self.metagraph.sync(subtensor=self.subtensor)
        if winner_hotkey not in self.metagraph.hotkeys:
            logger.warning(f"Winner hotkey not found in metagraph: {winner_hotkey}")
            return
        winner_uid = self.metagraph.hotkeys.index(winner_hotkey)

        weight_by_uid: dict[int, float] = {
            winner_uid: (1.0 - burn_ratio),
            self.burn_uid: burn_ratio,
        }
        if winner_uid == self.burn_uid:
            weight_by_uid[winner_uid] = 1.0

        total = sum(weight_by_uid.values())
        if total <= 0:
            logger.warning("Computed zero total weight; skipping")
            return
        if abs(total - 1.0) > 1e-9:
            weight_by_uid = {uid: value / total for uid, value in weight_by_uid.items()}

        uids = np.array(list(weight_by_uid.keys()), dtype=np.int64)
        weights = np.array(list(weight_by_uid.values()), dtype=np.float32)
        logger.info("Setting weights")
        logger.info(f"Winner: {winner_hotkey}")
        logger.info(f"None-zero uids: {weight_by_uid.keys()}")
        logger.info(f"None-zero weights: {weight_by_uid.values()}")
        
        response = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.netuid,
            uids=uids,
            weights=weights,
            wait_for_inclusion=True,
            wait_for_finalization=False,
        )
        if response.success != True:
            logger.warning("set_weights() returned failure")

    def _set_burn_only_weights(self) -> None:

        logger.info("Setting weights")
        response = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.netuid,
            uids=np.array([self.burn_uid], dtype=np.int64),
            weights=np.array([1.0], dtype=np.float32),
            wait_for_inclusion=True,
            wait_for_finalization=False,
        )
        if response.success != True:
            logger.warning("set_weights() returned failure")

    def run(self) -> None:
        last_cycle_block = -1
        try:
            while True:
                current_block = self._sync_and_get_current_block()
                if (
                    last_cycle_block >= 0
                    and current_block - last_cycle_block < INTERVAL_BLOCKS
                ):
                    # time.sleep(12)
                    continue

                try:
                    self._submission_update_step()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"submission_update_step error: {exc}")

                try:
                    self._weight_setting_step()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"weight_setting_step error: {exc}")

                last_cycle_block = current_block
                # time.sleep(12)
        except KeyboardInterrupt:
            logger.info("Validator stopped by user")


@click.command()
def main() -> None:
    Validator().run()


if __name__ == "__main__":
    main()
