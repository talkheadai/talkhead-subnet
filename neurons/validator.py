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
from utils.logging import maybe_reset_wandb
from utils.git import check_and_update_code

import wandb
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

INTERVAL_BLOCKS = 180


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
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"Validator is not registered in metagraph: {self.wallet.hotkey.ss58_address}"
            )
            exit()
            
        self.config = replace(cfg, full_path=os.getcwd())
        # Each miner gets a unique identity (UID) in the network for differentiation.
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        self.burn_uid = self.metagraph.hotkeys.index(
            self.subtensor.subnet(netuid=self.netuid).owner_hotkey
        )
        if self.burn_uid < 0 or self.burn_uid >= len(self.metagraph.hotkeys):
            bt.logging.warning(f"Burn UID out of range: {self.burn_uid}")
            self.burn_uid = 0
        self._winner_uid: int | None = None
        self._metrics_etag: str | None = ""
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
        except Exception as e:
            bt.logging.error(f"Failed to initialize W&B run: {e}")
            self.config.wandb.off = True
            return

        self._wandb_start_date = datetime.now(timezone.utc).date()

        # Sign the run to ensure it's from the correct hotkey
        signature = self.wallet.hotkey.sign(run_id.encode()).hex()
        self.config.signature = signature
        wandb.config.update(self.config, allow_val_change=True)

        bt.logging.success(f"Started wandb run {run_name}")

    @staticmethod
    def _pick_winner(metrics_list: list[dict]) -> tuple[str, float] | None:
        valid: list[tuple[str, float]] = []
        for row in metrics_list:
            if not isinstance(row, dict):
                continue
            hotkey = row.get("hotkey")
            metrics = row.get("metrics")
            if not isinstance(hotkey, str) or not isinstance(metrics, dict):
                continue
            if metrics.get("error") != None:
                continue
            try:
                final_score = float(metrics.get("final_score"))
                if final_score == 0.0:
                    continue
            except (TypeError, ValueError):
                continue
            if not math.isfinite(final_score):
                continue
            valid.append((hotkey, final_score))

        if not valid or len(valid) == 0:
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
            bt.logging.warning(
                f"Failed to fetch submissions (status={status}): {submissions}"
            )
            return
        if len(submissions) == 0:
            bt.logging.info("No submissions received; skipping update")
            return

        exec_status, exec_response = _http_json(
            f"{self.executor_url.rstrip('/')}/update",
            "POST",
            headers=signed_subnet_headers(self.wallet, "/update"),
            body=submissions,
        )
        if not (200 <= exec_status < 300):
            bt.logging.warning(
                f"Executor update failed (status={exec_status}): {exec_response}"
            )
        else:
            bt.logging.debug(
                f"Executor update successful (status={exec_status}): {exec_response}"
            )

    def _weight_setting_step(self) -> None:
        keep_latest_weights = False
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
                bt.logging.warning(
                    "Metrics 304 Not Modified but no cached rows; burning all"
                )
                self._set_burn_only_weights()
                return
            bt.logging.warning("Metrics unchanged (304); keep the latest weights")
            keep_latest_weights = True
        elif 200 <= metric_status < 300:
            self._metrics_etag = resp_headers.get("etag") or None
            metrics_list = metrics_payload
            if isinstance(metrics_list, list):
                self._metrics_cache = metrics_list
        else:
            metrics_list = metrics_payload

        if not keep_latest_weights:
            if not isinstance(metrics_list, list) or len(metrics_list) == 0:
                bt.logging.info("No metrics available; burning all")
                self._set_burn_only_weights()
                return
            self.do_logging(metrics_list)
            winner = self._pick_winner(metrics_list)
            if winner is None:
                bt.logging.warning("All metrics invalid; burning all")
                self._set_burn_only_weights()
                return
            winner_hotkey, _ = winner

            self.metagraph.sync(subtensor=self.subtensor)
            if winner_hotkey not in self.metagraph.hotkeys:
                bt.logging.warning(f"Winner hotkey not found in metagraph: {winner_hotkey} | Burning all")
                self._set_burn_only_weights()
                return

            self._winner_uid = self.metagraph.hotkeys.index(winner_hotkey)
            bt.logging.info(f"🏆 Winner is Miner {self._winner_uid} | {winner_hotkey}")
        
        if self._winner_uid is None:
            bt.logging.warning("No winner found; burning all")
            self._set_burn_only_weights()
            return
            
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
            bt.logging.warning(
                f"Failed to fetch burn ratio (status={burn_status}): {burn_response}"
            )

        try:
            burn_ratio_value = burn_response.get("burn_ratio")
            burn_ratio = max(0.0, min(1.0, float(burn_ratio_value)))
        except Exception as e:
            bt.logging.warning(f"Failed to parse burn ratio: {e}")
            burn_ratio = 1.0

        weight_by_uid: dict[int, float] = {
            self._winner_uid: (1.0 - burn_ratio),
            self.burn_uid: burn_ratio,
        }
        if self._winner_uid == self.burn_uid:
            weight_by_uid[self._winner_uid] = 1.0

        total = sum(weight_by_uid.values())
        if total <= 0:
            bt.logging.warning("Computed zero total weight; burning all")
            self._set_burn_only_weights()
            return
        if abs(total - 1.0) > 1e-9:
            weight_by_uid = {uid: value / total for uid, value in weight_by_uid.items()}

        uids = np.array(list(weight_by_uid.keys()), dtype=np.int64)
        weights = np.array(list(weight_by_uid.values()), dtype=np.float32)
        bt.logging.debug("Setting weights")
        bt.logging.debug(f"None-zero uids: {weight_by_uid.keys()}")
        bt.logging.debug(f"None-zero weights: {weight_by_uid.values()}")

        response = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.netuid,
            uids=uids,
            weights=weights,
            wait_for_inclusion=True,
            wait_for_finalization=False,
        )
        if response.success != True:
            bt.logging.warning("set_weights() returned failure")

    def _set_burn_only_weights(self) -> None:
        self._winner_uid = None
        bt.logging.info("Setting burn only weights")
        response = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.netuid,
            uids=np.array([self.burn_uid], dtype=np.int64),
            weights=np.array([1.0], dtype=np.float32),
            wait_for_inclusion=True,
            wait_for_finalization=False,
        )
        if response.success != True:
            bt.logging.warning("set_weights() returned failure")

    def do_logging(
        self,
        metrics_list: list[dict],
    ):
        for metric in metrics_list:
            hotkey = metric.get("hotkey")
            try:
                uid = self.metagraph.hotkeys.index(hotkey)
            except ValueError:
                continue
            metrics = metric.get("metrics")
            if metrics is not None:
                if metrics.get("error") is not None:
                    bt.logging.warning(
                        f"Invalid submission for hotkey {hotkey}: {metrics.get('error')}"
                    )
                    continue
                efficiency = metrics.get("efficiency", {})
                try:
                    final_score = round(metrics.get("final_score", 0.0), 2)
                    quality_score = round(metrics.get("quality_score", 0.0), 2)
                    peak_vram_gb = round(efficiency.get("peak_vram_gb", 0.0), 2)
                    inference_time_sec = round(efficiency.get("inference_time_sec", 0.0), 2)
                except Exception as e:
                    bt.logging.warning(f"Failed to parse metrics for hotkey {hotkey}: {e}")
                    continue
                bt.logging.info(
                    f"Metrics for Miner {uid} | {hotkey}: Final Score {final_score} | Quality Score {quality_score} | Peak VRAM {peak_vram_gb} | Inference Time {inference_time_sec}"
                )
                if not self.config.wandb.off:
                    wandb.log(
                        {
                            f"miner_{uid}_{hotkey}_final_score": final_score,
                            f"miner_{uid}_{hotkey}_quality_score": quality_score,
                            f"miner_{uid}_{hotkey}_peak_vram_gb": peak_vram_gb,
                            f"miner_{uid}_{hotkey}_inference_time_sec": inference_time_sec,
                        }
                    )

    def run(self) -> None:
        last_cycle_block = -1
        try:
            while True:
                current_block = self._sync_and_get_current_block()
                if (
                    last_cycle_block >= 0
                    and current_block - last_cycle_block < INTERVAL_BLOCKS
                ):
                    bt.logging.debug(f"Validator is running...")
                    time.sleep(12 * 10) # Wating for 10 blocks
                    continue
                
                if self.subtensor.network != "test":
                    check_and_update_code()

                try:
                    self._submission_update_step()
                except Exception as exc:  # noqa: BLE001
                    bt.logging.warning(f"submission_update_step error: {exc}")

                try:
                    self._weight_setting_step()
                except Exception as exc:  # noqa: BLE001
                    bt.logging.warning(f"weight_setting_step error: {exc}")

                last_cycle_block = current_block

                # prevent W&B logs from becoming massive
                maybe_reset_wandb(self)
        except KeyboardInterrupt:
            bt.logging.info("Validator stopped by user")


def main() -> None:
    cfg = load_app_config(config())
    Validator(cfg).run()


if __name__ == "__main__":
    main()
