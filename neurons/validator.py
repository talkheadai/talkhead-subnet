from __future__ import annotations

import bittensor as bt
import json
import logging
import math
import os
import time
from dataclasses import replace
from urllib import error, request

import numpy as np

from config import AppConfig, config, load_app_config
from utils import __version__
from utils.sign import signed_subnet_headers

import wandb
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
INTERVAL_BLOCKS = 360

def _header_dict(msg: object) -> dict[str, str]:
    return {k.lower(): v for k, v in msg.items()}


def _http_json(
    url: str,
    method: str,
    body: object | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> tuple[int, object]:
    status, payload, _hdrs = _http_json_with_headers(
        url, method, body=body, headers=headers, timeout=timeout
    )
    return status, payload


def _http_json_with_headers(
    url: str,
    method: str,
    body: object | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> tuple[int, object, dict[str, str]]:
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
            hdrs = _header_dict(response.headers)
            return response.status, json.loads(raw) if raw else None, hdrs
    except error.HTTPError as http_err:
        hdrs = _header_dict(http_err.headers)
        raw = http_err.read().decode("utf-8")
        if http_err.code == 304:
            return 304, None, hdrs
        try:
            return http_err.code, json.loads(raw) if raw else None, hdrs
        except json.JSONDecodeError:
            return http_err.code, raw, hdrs
    except Exception as exc:  # noqa: BLE001
        return 0, str(exc), {}


class Validator:
    def __init__(self, cfg: AppConfig) -> None:

        self._app_cfg = cfg
        self.netuid = cfg.netuid
        self.subnet_api_url = cfg.subnet_api_url
        self.executor_url = cfg.executor_url
        if not self.executor_url:
            self.executor_url = self.subnet_api_url
        self.wallet = bt.Wallet(name=cfg.wallet_name, hotkey=cfg.wallet_hotkey)
        self.subtensor = bt.Subtensor(network=cfg.network)
        self.metagraph = bt.Metagraph(netuid=self.netuid, network=cfg.network)
        self.config = replace(cfg, full_path=os.getcwd())
        # Each miner gets a unique identity (UID) in the network for differentiation.
        self.uid = self.metagraph.hotkeys.index(
            self.wallet.hotkey.ss58_address
        )
        self.burn_uid = self.metagraph.hotkeys.index(
            self.subtensor.subnet(netuid=self.netuid).owner_hotkey
        )
        if self.burn_uid < 0 or self.burn_uid >= len(self.metagraph.hotkeys):
            logger.warning(f"Burn UID out of range: {self.burn_uid}")
            self.burn_uid = 0
        self._metrics_etag: str | None = None
        self._metrics_cache: list | None = None
        self.init_wandb()

    def init_wandb(self) -> None:
        if self.config.wandb.off:
            return

        run_name = f"validator-{self.uid}-{__version__}"
        wandb_project = (
            self.config.wandb.project_name
            if self.subtensor.network != "test"
            else self.config.wandb.testnet_project_name
        )

        # Initialize the wandb run for the single project
        bt.logging.info(
            f"Initializing W&B run for '{self.config.wandb.entity}/{wandb_project}'"
        )
        try:
            run_id = wandb.init(
                name=run_name,
                project=wandb_project,
                entity=self.config.wandb.entity or None,
                config=self.config,
                dir=self.config.full_path or None,
                mode="offline" if self.config.wandb.offline else None,
            ).id
        except wandb.UsageError as e:
            bt.logging.warning(e)
            bt.logging.warning("Did you run wandb login?")
            return

        self._wandb_start_date = datetime.now(timezone.utc).date()

        # Sign the run to ensure it's from the correct hotkey
        signature = self.wallet.hotkey.sign(run_id.encode()).hex()
        self.config.signature = signature
        wandb.config.update(self.config, allow_val_change=True)

        bt.logging.success(f"Started wandb run {run_name}")

    @staticmethod
    def _pick_winner(metrics_list: list[dict]) -> tuple[str, float] | None:
        logger.info(f"Picking winner from {len(metrics_list)} metrics")
        valid: list[tuple[str, float]] = []
        for row in metrics_list:
            if not isinstance(row, dict):
                continue
            hotkey = row.get("hotkey")
            metrics = row.get("metrics")
            if not isinstance(hotkey, str) or not isinstance(metrics, dict):
                continue
            try:
                value = float(metrics.get("final_score"))
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
        status, submissions = _http_json(
            f"{self.subnet_api_url.rstrip('/')}/submissions",
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
        else:
            logger.info(f"Executor update successful (status={exec_status}): {exec_response}")

    def _weight_setting_step(self) -> None:
        req_headers = dict(signed_subnet_headers(self.wallet, "/metrics"))
        if self._metrics_etag:
            req_headers["If-None-Match"] = self._metrics_etag

        metric_status, metrics_payload, resp_headers = _http_json_with_headers(
            f"{self.executor_url.rstrip('/')}/metrics",
            "GET",
            headers=req_headers,
        )

        if metric_status == 304:
            if self._metrics_cache is None:
                logger.warning(
                    "Metrics 304 Not Modified but no cached rows; burning all"
                )
                self._set_burn_only_weights()
                return
            logger.info("Metrics unchanged (304); skipping weight setting")
            return
        elif 200 <= metric_status < 300:
            self._metrics_etag = resp_headers.get("etag") or None
            metrics_list = metrics_payload
            if isinstance(metrics_list, list):
                self._metrics_cache = metrics_list
        else:
            metrics_list = metrics_payload

        if not isinstance(metrics_list, list) or len(metrics_list) == 0:
            logger.info("No metrics available; burning all")
            self._set_burn_only_weights()
            return

        self.do_wandb_logging(metrics_list)
        winner = self._pick_winner(metrics_list)
        if winner is None:
            logger.info("All metrics invalid; skipping weight setting")
            return
        winner_hotkey, _winner_score = winner

        burn_ratio = 1.0
        burn_status, burn_response = _http_json(
            f"{self.subnet_api_url.rstrip('/')}/burn_ratio",
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
            logger.info(f"Fetched burn ratio: {burn_ratio}")
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

        logger.info("Setting burn only weights")
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


    def do_wandb_logging(
            self, 
            metrics_list: list[dict],
        ):
        if self.config.wandb.off:
            return

        # total_uids = [row.get("uid") for row in metrics_list]
        # total_composites = [row.get("composite") for row in metrics_list]
        # total_latency_scores = [row.get("latency_score") for row in metrics_list]
        # applied_rewards = [row.get("applied_reward") for row in metrics_list]
        # detailed_metrics = [row.get("detailed_metric") for row in metrics_list]

        # uid_to_hotkey = {uid: self.metagraph.hotkeys[uid] for uid in total_uids}
        # for uid, composite, latency_score, applied_reward, detailed_metric in zip(total_uids, total_composites, total_latency_scores, applied_rewards, detailed_metrics):
        #     wandb.log(
        #         {
        #             f"miner_{uid}_{uid_to_hotkey[uid]}_composite": composite,
        #             f"miner_{uid}_{uid_to_hotkey[uid]}_latency_score": latency_score,
        #             f"miner_{uid}_{uid_to_hotkey[uid]}_reward": applied_reward,
        #             f"miner_{uid}_{uid_to_hotkey[uid]}_syncnet": detailed_metric.get("S_syncnet", 0.0),
        #             f"miner_{uid}_{uid_to_hotkey[uid]}_arcface": detailed_metric.get("S_arcface", 0.0),
        #             f"miner_{uid}_{uid_to_hotkey[uid]}_quality": detailed_metric.get("S_quality", 0.0),
        #             f"miner_{uid}_{uid_to_hotkey[uid]}_blink": detailed_metric.get("S_blink", 0.0),
        #             f"miner_{uid}_{uid_to_hotkey[uid]}_reason": detailed_metric.get("reason", ""),
        #         },
        #     )

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


def main() -> None:
    bt_cfg = config()
    cfg = load_app_config(bt_cfg)
    logger.info(f"Validator config: {cfg}")
    Validator(cfg).run()


if __name__ == "__main__":
    main()
