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
import copy
import numpy as np
import asyncio
import argparse
import threading
import requests
import bittensor as bt
from talkhead.constants import TALKHEAD_SERVER_HOST, TESTNET_TALKHEAD_SERVER_HOST

from typing import List, Union
from traceback import print_exception

from talkhead.base.neuron import BaseNeuron
from talkhead.base.utils.weight_utils import (
    process_weights_for_netuid,
    convert_weights_and_uids_for_emit,
)  # TODO: Replace when bittensor switches to numpy
from talkhead.mock import MockDendrite
from talkhead.utils.config import add_validator_args
from talkhead.utils.git import check_and_update_code

class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    neuron_type: str = "ValidatorNeuron"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_validator_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        # Dendrite lets us send messages to other nodes (axons) in the network.
        if self.config.mock:
            self.dendrite = MockDendrite(wallet=self.wallet)
        else:
            self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        self.scores = np.zeros(self.metagraph.n, dtype=np.float32)

        # Init sync with the network. Updates the metagraph.
        self.sync()

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            bt.logging.warning("axon off, not serving ip to chain.")

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: Union[threading.Thread, None] = None
        self.lock = asyncio.Lock()

        # Burn configuration (defaults loaded from environment; refreshed via server)
        self.burn_ratio = 1.0
        self.burn_uid = 0
        self.top_miner_cap = 2
        self.decay_rate = 0.05
        self.blend_factor = 0.7
        self._last_param_refresh_time = 0.0
        self._param_refresh_interval = 300  # seconds

    def build_signed_headers(self, path: str) -> dict:
        """
        Build request headers containing the validator hotkey, timestamp, and a
        signature so the TALKHEAD server can authenticate the caller.
        """
        timestamp = str(int(time.time()))
        message = f"<Bytes>{path}:{timestamp}:{self.wallet.hotkey.ss58_address}</Bytes>"

        try:
            signature = self.wallet.hotkey.sign(data=message)
        except Exception as exc:
            bt.logging.error(f"Failed to sign request for {path}: {exc}")
            raise

        return {
            "X-Validator-Hotkey": self.wallet.hotkey.ss58_address,
            "X-Validator-Timestamp": timestamp,
            "X-Validator-Signature": signature.hex(),
        }

    def serve_axon(self):
        """Serve axon to enable external connections."""

        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.axon(wallet=self.wallet, config=self.config)

            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
                bt.logging.info(
                    f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
                )
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {e}")
                pass

        except Exception as e:
            bt.logging.error(
                f"Failed to create Axon initialize with exception: {e}"
            )
            pass

    async def concurrent_forward(self):
        coroutines = [
            self.forward()
            for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines)

    def run(self, propagate_exceptions: bool = False):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Continuously forwards queries to the miners on the network, rewarding their responses and updating the scores accordingly.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The essence of the validator's operations is in the forward function, which is called every step. The forward function is responsible for querying the network and scoring the responses.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that validator is registered on the network.
        self.sync()

        bt.logging.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                if self.subtensor.network == "finney":
                    check_and_update_code()
                bt.logging.info(f"step({self.step}) block({self.block})")

                # Run multiple forwards concurrently.
                self.loop.run_until_complete(self.concurrent_forward())

                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                self.sync()

                bt.logging.info(f"Sleeping for 60 seconds before next forward call")
                time.sleep(60)

                self.step += 1

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            bt.logging.error(f"Error during validation: {str(err)}")
            bt.logging.debug(
                str(print_exception(type(err), err, err.__traceback__))
            )
            if propagate_exceptions:
                raise

    def _run_thread_entry(self):
        """
        Entry point for the background thread. Restarts the run loop on failure.
        """
        restart_delay = 5  # seconds
        while not self.should_exit:
            try:
                # Propagate so we can restart on failures.
                self.run(propagate_exceptions=True)
                # run exits normally only when should_exit is set; stop restarting.
                break
            except Exception as err:
                bt.logging.error(
                    f"Validator run crashed; restarting in {restart_delay}s: {err}"
                )
                time.sleep(restart_delay)

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(
                target=self._run_thread_entry, daemon=True
            )
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def refresh_parameters(self, force: bool = False):
        """
        Refresh parameters from TALKHEAD_SERVER_HOST/params with caching.
        """
        now = time.time()
        if not force and (now - self._last_param_refresh_time) < self._param_refresh_interval:
            return

        host = TALKHEAD_SERVER_HOST if self.subtensor.network == "finney" else TESTNET_TALKHEAD_SERVER_HOST

        try:
            headers = self.build_signed_headers(f"{host}/params")
            response = requests.get(f"{host}/params", headers=headers, timeout=5)
            response.raise_for_status()
            payload = response.json() or {}
        except Exception as exc:
            bt.logging.warning(f"Unable to refresh parameters from server, using cached defaults. Error: {exc}")
            self._last_param_refresh_time = now
            return

        updated = False
        def update_param(attr, cast, invalid_msg_fn=None):
            nonlocal updated
            raw = payload.get(attr)
            if raw is None:
                return
            try:
                value = cast(raw)
            except (TypeError, ValueError):
                msg = invalid_msg_fn(raw) if invalid_msg_fn else f"Ignoring invalid {attr}: {raw}"
                bt.logging.warning(msg)
                return
            if getattr(self, attr) != value:
                updated = True
            setattr(self, attr, value)

        update_param("burn_ratio", float, invalid_msg_fn=lambda raw: f"Ignoring non-numeric burn_ratio: {raw}",)
        update_param("burn_uid", int, invalid_msg_fn=lambda raw: f"Ignoring non-integer burn_uid: {raw}",)
        update_param("top_miner_cap", int, invalid_msg_fn=lambda raw: f"Ignoring non-integer top_miner_cap: {raw}",)
        update_param("decay_rate", float, invalid_msg_fn=lambda raw: f"Ignoring non-numeric decay_rate: {raw}",)
        update_param("blend_factor", float, invalid_msg_fn=lambda raw: f"Ignoring non-numeric blend_factor: {raw}",)

        if updated:
            bt.logging.info(f"Updated parameters from server: ratio={self.burn_ratio}, uid={self.burn_uid}, top_miner_cap={self.top_miner_cap}, decay_rate={self.decay_rate}, blend_factor={self.blend_factor}")

        self._last_param_refresh_time = now

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """

        # Check if self.scores contains any NaN values and log a warning if it does.
        if np.isnan(self.scores).any():
            bt.logging.warning(
                f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        # Compute the norm of the scores
        norm = np.linalg.norm(self.scores, ord=1, axis=0, keepdims=True)

        # Check if the norm is zero or contains NaN values
        if np.any(norm == 0) or np.isnan(norm).any():
            norm = np.ones_like(norm)  # Avoid division by zero or NaN

        # Compute raw_weights safely
        raw_weights = self.scores / norm

        bt.logging.debug("raw_weights", raw_weights)
        bt.logging.debug("raw_weight_uids", str(self.metagraph.uids.tolist()))

        # Ensure burn config is up to date before applying it.
        self.refresh_parameters()
        burn_ratio = self.burn_ratio
        burn_uid = self.burn_uid

        if burn_ratio > 0:
            if burn_uid < 0 or burn_uid >= self.metagraph.n:
                bt.logging.warning(f"Burn UID {burn_uid} is outside the metagraph range; skipping burn application.")
            else:
                bt.logging.info(f"Burning {burn_ratio} of emissions to UID {burn_uid}")
                burn_weights = np.zeros(self.metagraph.n, dtype=np.float32)
                burn_weights[burn_uid] = burn_ratio
                raw_weights = burn_weights + raw_weights * (1 - burn_ratio)

        bt.logging.debug("raw_weights after burn", raw_weights)

        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=raw_weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        bt.logging.debug("processed_weights", processed_weights)
        bt.logging.debug("processed_weight_uids", processed_weight_uids)

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )
        bt.logging.debug("uint_weights", uint_weights)
        bt.logging.debug("uint_uids", uint_uids)

        # Set the weights on chain via our subtensor connection.
        result, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=False,
            wait_for_inclusion=False,
            version_key=self.spec_version,
        )
        if result is True:
            bt.logging.info("set_weights on chain successfully!")
        else:
            bt.logging.error("set_weights failed", msg)

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_axons = list(self.metagraph.axons)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_axons == list(self.metagraph.axons):
            return

        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = np.zeros((self.metagraph.n))
            min_len = min(len(self.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average

        # Update the hotkeys.
        self.hotkeys = list(self.metagraph.hotkeys)

    def update_scores(self, rewards: np.ndarray, uids: np.ndarray):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        # Check if rewards contains NaN values.
        if np.isnan(rewards).any():
            bt.logging.warning(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = np.nan_to_num(rewards, nan=0)

        # Ensure rewards is a numpy array.
        rewards = np.asarray(rewards)

        # Check if `uids` is already a numpy array and copy it to avoid the warning.
        if isinstance(uids, np.ndarray):
            uids_array = uids.copy()
        else:
            uids_array = np.array(uids)

        # Handle edge case: If either rewards or uids_array is empty.
        if rewards.size == 0 or uids_array.size == 0:
            bt.logging.info(f"rewards: {rewards}, uids_array: {uids_array}")
            bt.logging.warning(
                "Either rewards or uids_array is empty. No updates will be performed."
            )
            return

        # Check if sizes of rewards and uids_array match.
        if rewards.size != uids_array.size:
            raise ValueError(
                f"Shape mismatch: rewards array of shape {rewards.shape} "
                f"cannot be broadcast to uids array of shape {uids_array.shape}"
            )

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        scattered_rewards: np.ndarray = np.zeros_like(self.scores)
        scattered_rewards[uids_array] = rewards
        bt.logging.debug(f"Scattered rewards: {rewards}")

        # Update scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        alpha: float = self.config.neuron.moving_average_alpha
        self.scores: np.ndarray = (
            alpha * scattered_rewards + (1 - alpha) * self.scores
        )
        bt.logging.debug(f"Updated moving avg scores: {self.scores}")

    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")

        # Save the state of the validator to file.
        np.savez(
            self.config.neuron.full_path + "/state.npz",
            step=self.step,
            scores=self.scores,
            hotkeys=self.hotkeys,
        )

    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")

        # Load the state of the validator from file.
        state = np.load(self.config.neuron.full_path + "/state.npz")
        self.step = state["step"]
        self.scores = state["scores"]
        self.hotkeys = state["hotkeys"]
