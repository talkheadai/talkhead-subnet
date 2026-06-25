import os
import sys
import time
from pathlib import Path
from typing import Tuple

import bittensor as bt

from config import config, load_app_config
from talkhead.protocol import ImageRef
from talkhead.constant import NETUID


def _resolve_image_ref(cli_image_ref: str | None, cfg_image_ref: str) -> str:
    if cli_image_ref is not None and cli_image_ref.strip():
        return cli_image_ref.strip()
    env_ref = os.getenv("IMAGE_REF", "").strip()
    if env_ref:
        return env_ref
    return (cfg_image_ref or "").strip()


class TalkHeadMiner:
    def __init__(self, bt_cfg: bt.Config, image_ref: str) -> None:
        self.image_ref = image_ref
        self.netuid = bt_cfg.netuid or NETUID
        self.wallet = bt.Wallet(config=bt_cfg)
        self.subtensor = bt.Subtensor(config=bt_cfg)
        self.metagraph = bt.Metagraph(netuid=self.netuid)
        self.axon = bt.Axon(wallet=self.wallet, config=bt_cfg)

    def forward(self, synapse: ImageRef) -> ImageRef:
        synapse.image_ref = self.image_ref
        return synapse

    def blacklist(self, synapse: ImageRef) -> Tuple[bool, str]:
        if synapse.dendrite is None or not synapse.dendrite.hotkey:
            return True, "Missing dendrite hotkey"
        return False, ""

    def run(self) -> None:
        self.metagraph.sync(subtensor=self.subtensor)
        hotkey = self.wallet.hotkey.ss58_address
        if hotkey not in self.metagraph.hotkeys:
            bt.logging.error(
                f"Miner is not registered on netuid {self.netuid}: {hotkey}"
            )
            sys.exit(1)

        bt.logging.info(f"Serving image_ref={self.image_ref} on axon")
        self.axon.attach(
            forward_fn=forward_image_ref,
            blacklist_fn=blacklist_image_ref,
        ).serve(
            netuid=self.netuid,
            subtensor=self.subtensor,
        ).start()

        bt.logging.success(
            f"Axon serving ImageRef on netuid {self.netuid} "
            f"(port={self.axon.port})"
        )

        try:
            while True:
                time.sleep(12)
        except KeyboardInterrupt:
            bt.logging.info("Miner stopped by user")
            self.axon.stop()


_miner: TalkHeadMiner | None = None


def forward_image_ref(synapse: ImageRef) -> ImageRef:
    if _miner is None:
        raise RuntimeError("TalkHead miner is not initialized")
    return _miner.forward(synapse)


def blacklist_image_ref(synapse: ImageRef) -> Tuple[bool, str]:
    if _miner is None:
        return True, "Miner not initialized"
    return _miner.blacklist(synapse)


def main() -> None:
    """
    TalkHead miner entrypoint.

    Serves the miner's Docker image digest on its axon via ``ImageRef``.
    Validators discover model registrations by querying the axon directly.
    """
    bt_cfg = config()
    app_cfg = load_app_config(bt_cfg)
    image_ref = _resolve_image_ref(getattr(bt_cfg, "image_ref", None), app_cfg.image_ref)
    if "@sha256:" not in image_ref:
        bt.logging.error("Error: image_ref must include '@sha256:'.")
        sys.exit(1)

    global _miner
    _miner = TalkHeadMiner(bt_cfg, image_ref)
    _miner.run()


if __name__ == "__main__":
    main()
