from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from urllib import error, request

import bittensor as bt
import numpy as np
import wandb
from datetime import datetime, timezone
from dotenv import load_dotenv

from config import AppConfig, config, load_app_config
from talkhead.protocol import ImageRef
from talkhead.constant import NETUID, BURN_UID, BURN_RATIO
from utils import __version__
from utils.sign import signed_subnet_headers
from utils.logging import maybe_reset_wandb
from utils.git import check_and_update_code

load_dotenv()

INTERVAL_BLOCKS = 180
MINER_QUERY_BATCH_SIZE = max(1, int(os.getenv("MINER_QUERY_BATCH_SIZE", "16")))
MINER_QUERY_TIMEOUT = float(os.getenv("MINER_QUERY_TIMEOUT", "12"))



class Validator:
    def __init__(self, cfg: AppConfig, bt_cfg: bt.Config) -> None:
        self._app_cfg = cfg
        self._bt_cfg = bt_cfg
        self.netuid = bt_cfg.netuid or NETUID
        self.wallet = bt.Wallet(config=bt_cfg)
        self.subtensor = bt.Subtensor(config=bt_cfg)
        self.metagraph = bt.Metagraph(netuid=self.netuid)
        self.dendrite = bt.Dendrite(wallet=self.wallet)
        self._query_batch_size = MINER_QUERY_BATCH_SIZE
        self._query_timeout = MINER_QUERY_TIMEOUT
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"Validator is not registered in metagraph: {self.wallet.hotkey.ss58_address}"
            )
            sys.exit(1)
            
        self.config = replace(cfg, full_path=os.getcwd())
        # Each miner gets a unique identity (UID) in the network for differentiation.
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        self.burn_uid = self.metagraph.hotkeys.index(
            self.subtensor.subnet(netuid=self.netuid).owner_hotkey
        )
        if self.burn_uid < 0 or self.burn_uid >= len(self.metagraph.hotkeys):
            bt.logging.warning(f"Burn UID out of range: {self.burn_uid}")
            self.burn_uid = BURN_UID
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

    def _serving_miner_targets(self) -> list[tuple[int, bt.AxonInfo]]:
        """Return (uid, axon) pairs for miners with a reachable axon endpoint."""
        targets: list[tuple[int, bt.AxonInfo]] = []
        for uid, axon in enumerate(self.metagraph.axons):
            if uid == self.uid:
                continue
            if not axon.is_serving:
                # bt.logging.debug(f"Skipping UID {uid}: axon not serving")
                continue
            targets.append((uid, axon))
        return targets

    def _parse_image_ref_response(
        self,
        uid: int,
        axon: bt.AxonInfo,
        response: object,
    ) -> dict[str, str] | None:
        """Validate a single ImageRef response and return an executor submission row."""
        hotkey = axon.hotkey
        if not isinstance(response, ImageRef):
            # bt.logging.warning(f"UID {uid} ({hotkey}): unexpected response type")
            return None

        if response.is_timeout:
            # bt.logging.warning(f"UID {uid} ({hotkey}): ImageRef query timed out")
            return None
        if response.is_failure:
            status = (
                response.dendrite.status_message
                if response.dendrite is not None
                else "unknown error"
            )
            # bt.logging.warning(f"UID {uid} ({hotkey}): ImageRef query failed ({status})")
            return None

        image_ref = (response.image_ref or "").strip()
        if "@sha256:" not in image_ref:
            bt.logging.warning(f"UID {uid} ({hotkey}): missing or invalid image_ref")
            return None

        bt.logging.debug(f"UID {uid} ({hotkey}): {image_ref}")
        return {"hotkey": hotkey, "image_ref": image_ref}

    async def collect_miner_digests(self) -> list[dict[str, str]]:
        """
        Query miner axons for ``ImageRef`` synapses and collect docker digests.

        Uses the metagraph to discover serving miners, queries axons in batches,
        and returns rows shaped for the executor ``/update`` endpoint:
        ``[{"hotkey": "...", "image_ref": "repo@sha256:..."}]``.
        """
        self.metagraph.sync(subtensor=self.subtensor)
        targets = self._serving_miner_targets()
        if not targets:
            bt.logging.info("No serving miners found in metagraph")
            return []

        submissions: list[dict[str, str]] = []
        batch_count = (len(targets) + self._query_batch_size - 1) // self._query_batch_size
        bt.logging.info(
            f"Collecting ImageRef digests from {len(targets)} miners "
            f"in {batch_count} batch(es)"
        )
        
        for batch_index in range(0, len(targets), self._query_batch_size):
            batch = targets[batch_index : batch_index + self._query_batch_size]
            uids = [uid for uid, _ in batch]
            axons = [axon for _, axon in batch]
            batch_num = batch_index // self._query_batch_size + 1

            bt.logging.debug(
                f"ImageRef batch {batch_num}/{batch_count}: querying UIDs {uids}"
            )

            try:
                responses = await self.dendrite.forward(
                    axons=axons,
                    synapse=ImageRef(),
                    timeout=self._query_timeout,
                    deserialize=False,
                )
            except Exception as exc:  # noqa: BLE001
                bt.logging.warning(
                    f"ImageRef batch {batch_num}/{batch_count} failed: {exc}"
                )
                continue

            if not isinstance(responses, list):
                responses = [responses]

            for uid, axon, response in zip(uids, axons, responses):
                row = self._parse_image_ref_response(uid, axon, response)
                if row is not None:
                    submissions.append(row)

        bt.logging.info(
            f"Collected {len(submissions)} digests from {len(targets)} serving miners"
        )
        return submissions

    def _submission_update_step(self) -> None:
        """
        Loop 1: query miners directly for ImageRef digests and forward them to the executor.
        """
        try:
            submissions = asyncio.run(self.collect_miner_digests())
        except Exception as exc:  # noqa: BLE001
            bt.logging.warning(f"collect_miner_digests error: {exc}")
            return



    def _weight_setting_step(self) -> None:
        """
        Loop 2: read executor metrics and set on-chain weights.
        """
        
        return
        metrics_list = self.get_metrics()

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
            
        weight_by_uid: dict[int, float] = {
            self._winner_uid: (1.0 - BURN_RATIO),
            self.burn_uid: BURN_RATIO,
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
                
                if self.subtensor.network != "test" and self.netuid == NETUID:
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
    bt_cfg = config()
    app_cfg = load_app_config(bt_cfg)
    Validator(app_cfg, bt_cfg).run()


if __name__ == "__main__":
    main()
