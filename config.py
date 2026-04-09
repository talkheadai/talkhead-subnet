import argparse
import os
import sys
from dataclasses import dataclass, field, replace
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Defaults for W&B (mirrored on CLI via add_args so --help and AppConfig stay aligned).
WANDB_PROJECT_NAME_DEFAULT = "talkhead-subnet"
WANDB_TESTNET_PROJECT_NAME_DEFAULT = "talkhead-testnet"
WANDB_ENTITY_DEFAULT = "talkhead-ai"


@dataclass
class WandbConfig:
    off: bool = True
    project_name: str = WANDB_PROJECT_NAME_DEFAULT
    testnet_project_name: str = WANDB_TESTNET_PROJECT_NAME_DEFAULT
    entity: str = WANDB_ENTITY_DEFAULT
    offline: bool = False


@dataclass
class AppConfig:
    image_ref: str = ""
    subnet_api_url: str = "https://subnet.talkhead.ai"
    executor_url: str = "http://localhost:9000"
    wallet_name: str = "default"
    wallet_hotkey: str = "default"
    network: str = "finney"
    netuid: int = 108
    full_path: str = ""
    wandb: WandbConfig = field(default_factory=WandbConfig)
    signature: str = ""


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _executor_url_from_env(subnet_api_url: str) -> str:
    raw = os.getenv("EXECUTOR_API_URL")
    if raw is None or str(raw).strip() == "":
        return subnet_api_url
    return str(raw)


def add_args(parser: argparse.ArgumentParser) -> None:
    """TalkHead-specific CLI flags (used together with bt.Wallet / Subtensor / Axon)."""
    parser.add_argument(
        "--image-ref",
        "--image_ref",
        type=str,
        default=None,
        help="Container image ref for miner (overrides IMAGE_REF)",
    )
    parser.add_argument(
        "--netuid",
        type=int,
        default=None,
        help="Subnet netuid (overrides NETUID)",
    )
    parser.add_argument(
        "--wandb.project_name",
        dest="wandb_project_name",
        type=str,
        default=WANDB_PROJECT_NAME_DEFAULT,
        help="Weights & Biases project name (mainnet / non-test)",
    )
    parser.add_argument(
        "--wandb.testnet_project_name",
        dest="wandb_testnet_project_name",
        type=str,
        default=WANDB_TESTNET_PROJECT_NAME_DEFAULT,
        help="Weights & Biases project name when subtensor.network is test",
    )
    parser.add_argument(
        "--wandb.entity",
        dest="wandb_entity",
        type=str,
        default=WANDB_ENTITY_DEFAULT,
        help="Weights & Biases entity (team)",
    )
    parser.add_argument(
        "--wandb.off",
        dest="wandb_off",
        action="store_true",
        default=False,
        help="Force W&B off (default: false)",
    )
    parser.add_argument(
        "--wandb.offline",
        dest="wandb_offline",
        action="store_true",
        help="Run W&B in offline mode (sets wandb.offline=true, default: false unless --wandb.offline)",
    )


def config(args: list[str] | None = None) -> Any:
    """Parse argv with Bittensor + TalkHead arguments; returns bt.Config (Munch-like)."""
    argv = sys.argv[:]
    try:
        if "--help" in sys.argv or "-h" in sys.argv:
            sys.argv = [sys.argv[0]]
        import bittensor as bt
    finally:
        sys.argv = argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    bt.Wallet.add_args(parser)
    bt.Subtensor.add_args(parser)
    bt.Axon.add_args(parser)
    add_args(parser)
    return bt.Config(parser=parser, args=args)


def load_config(
    *,
    image_ref: str | None = None,
    wallet_name: str | None = None,
    wallet_hotkey: str | None = None,
    network: str | None = None,
    netuid: int | None = None,
    wandb_project_name: str | None = None,
    wandb_testnet_project_name: str | None = None,
    wandb_entity: str | None = None,
    wandb_off: bool | None = None,
    wandb_offline: bool | None = None,
    signature: str | None = None,
) -> AppConfig:
    subnet_api_url = str(os.getenv("SUBNET_API_URL", "https://subnet.talkhead.ai"))

    img = str(os.getenv("IMAGE_REF", ""))

    cfg = AppConfig(
        image_ref=img,
        subnet_api_url=subnet_api_url,
        executor_url=_executor_url_from_env(subnet_api_url),
        wallet_name=str(os.getenv("WALLET_NAME", "default")),
        wallet_hotkey=str(os.getenv("HOTKEY_NAME", "default")),
        network=str(os.getenv("NETWORK", "finney")),
        netuid=_as_int(os.getenv("NETUID", "108"), 108),
        wandb=WandbConfig(),
    )

    overrides: dict[str, Any] = {}
    if image_ref is not None:
        overrides["image_ref"] = str(image_ref)
    if wallet_name is not None:
        overrides["wallet_name"] = str(wallet_name)
    if wallet_hotkey is not None:
        overrides["wallet_hotkey"] = str(wallet_hotkey)
    if network is not None:
        overrides["network"] = str(network)
    if netuid is not None:
        overrides["netuid"] = int(netuid)

    cfg = replace(cfg, **overrides)

    wandb_overrides: dict[str, Any] = {}
    if wandb_project_name is not None:
        wandb_overrides["project_name"] = str(wandb_project_name)
    if wandb_testnet_project_name is not None:
        wandb_overrides["testnet_project_name"] = str(wandb_testnet_project_name)
    if wandb_entity is not None:
        wandb_overrides["entity"] = str(wandb_entity)
    if wandb_off is not None:
        wandb_overrides["off"] = bool(wandb_off)
    if wandb_offline is not None:
        wandb_overrides["offline"] = bool(wandb_offline)
    if wandb_overrides:
        cfg = replace(cfg, wandb=replace(cfg.wandb, **wandb_overrides))

    if not cfg.executor_url.strip():
        cfg = replace(cfg, executor_url=cfg.subnet_api_url)
    return cfg


def _bt_cli_was_set(bt_cfg: Any, key: str) -> bool:
    """True if this flag was passed on the command line (not just parser defaults)."""
    is_set = getattr(bt_cfg, "_Config__is_set", None)
    if is_set is None:
        return False
    return bool(is_set.get(key))


def load_app_config(bt_cfg: Any) -> AppConfig:
    """Merge bt.Config with env: only CLI-given flags override; else env defaults for the rest."""
    return load_config(
        image_ref=getattr(bt_cfg, "image_ref", None)
        if _bt_cli_was_set(bt_cfg, "image_ref")
        else None,
        wallet_name=bt_cfg.wallet.name
        if _bt_cli_was_set(bt_cfg, "wallet.name")
        else None,
        wallet_hotkey=bt_cfg.wallet.hotkey
        if _bt_cli_was_set(bt_cfg, "wallet.hotkey")
        else None,
        network=bt_cfg.subtensor.network
        if _bt_cli_was_set(bt_cfg, "subtensor.network")
        else None,
        netuid=getattr(bt_cfg, "netuid", None)
        if _bt_cli_was_set(bt_cfg, "netuid")
        else None,
        wandb_project_name=str(
            getattr(bt_cfg, "wandb_project_name", WANDB_PROJECT_NAME_DEFAULT)
        ),
        wandb_testnet_project_name=str(
            getattr(bt_cfg, "wandb_testnet_project_name", WANDB_TESTNET_PROJECT_NAME_DEFAULT)
        ),
        wandb_entity=str(getattr(bt_cfg, "wandb_entity", WANDB_ENTITY_DEFAULT)),
        wandb_off=True if getattr(bt_cfg, "wandb_off", None) else False,
        wandb_offline=True if _bt_cli_was_set(bt_cfg, "wandb_offline") else None,
    )
