import json
import os
import sys
from urllib import error, request

import bittensor as bt
from config import config, load_app_config
from utils.sign import signed_subnet_headers


def _post_json(url: str, body: dict, headers: dict[str, str], timeout: int = 30) -> tuple[int, str]:
    payload = json.dumps(body).encode("utf-8")
    req = request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    for key, value in headers.items():
        req.add_header(key, value)

    try:
        with request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8")
    except error.HTTPError as http_err:
        return http_err.code, http_err.read().decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        return 0, str(exc)


def _resolve_image_ref(cli_image_ref: str | None, cfg_image_ref: str) -> str:
    if cli_image_ref is not None and cli_image_ref.strip():
        return cli_image_ref.strip()
    env_ref = os.getenv("IMAGE_REF", "").strip()
    if env_ref:
        return env_ref
    return (cfg_image_ref or "").strip()


def main() -> None:
    """
    TalkHead miner entrypoint.
    This miner is submission-only and does not serve axon requests.
    """
    bt_cfg = config()
    cfg = load_app_config(bt_cfg)
    image_ref = _resolve_image_ref(getattr(bt_cfg, "image_ref", None), cfg.image_ref)
    if "@sha256:" not in image_ref:
        bt.logging.error("Error: image_ref must include '@sha256:'.")
        sys.exit(1)

    wallet = bt.Wallet(name=cfg.wallet_name, hotkey=cfg.wallet_hotkey)

    bt.logging.info("Submitting image image_ref")
    body = {"hotkey": wallet.hotkey.ss58_address, "image_ref": image_ref}
    headers = signed_subnet_headers(wallet, "/submit")

    status, response = _post_json(f"{cfg.subnet_api_url.rstrip('/')}/submit", body, headers)
    if 200 <= status < 300:
        bt.logging.success(f"Success: {response}")
        return

    bt.logging.error(f"Error ({status}): {response}")
    sys.exit(1)


if __name__ == "__main__":
    main()
