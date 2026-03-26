import os
from dataclasses import dataclass
from typing import Any
import yaml
from dotenv import load_dotenv
load_dotenv()


@dataclass
class AppConfig:
    image_ref: str = ""
    subnet_api_url: str = "https://subnet.talkhead.ai"
    executor_url: str = "http://localhost:9000"
    wallet_name: str = "default"
    wallet_hotkey: str = "default"
    network: str = "finney"
    netuid: int = 108


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_config(path: str = "config.yaml") -> AppConfig:
    data: dict[str, Any] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
            if isinstance(loaded, dict):
                data = loaded

    return AppConfig(
        # Keep config.yaml miner-only: image_ref is read from YAML.
        image_ref=str(data.get("image_ref", "")),
        subnet_api_url=str(os.getenv("SUBNET_API_URL", "https://subnet.talkhead.ai")),
        executor_url=str(os.getenv("EXECUTOR_API_URL", "http://localhost:9000")),
        wallet_name=str(os.getenv("WALLET_NAME", "default")),
        wallet_hotkey=str(os.getenv("HOTKEY_NAME", "default")),
        network=str(os.getenv("NETWORK", "finney")),
        netuid=_as_int(os.getenv("NETUID", "108"), 108),
    )
