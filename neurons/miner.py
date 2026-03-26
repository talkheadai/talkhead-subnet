import click
import json
import os
import sys
from datetime import datetime, timezone
from urllib import error, request

import bittensor as bt

from config import load_config
from utils import signed_subnet_headers


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


@click.command()
@click.option("--image", default="", help="Container image ref (must include @sha256:...)")
@click.option("--config-path", default="config.yaml", help="Path to config.yaml")
def main(image: str, config_path: str) -> None:
    """
    TalkHead miner entrypoint.
    This miner is submission-only and does not serve axon requests.
    """
    cfg = load_config(config_path)
    image_ref = image or os.getenv("IMAGE_REF", "")
    if "@sha256:" not in image_ref:
        click.echo("Error: image image_ref must include '@sha256:'.")
        sys.exit(1)

    wallet = bt.Wallet(name=cfg.wallet_name, hotkey=cfg.wallet_hotkey)
    
    

    click.echo("Submitting image image_ref")
    body = {"hotkey": wallet.hotkey.ss58_address, "image_ref": image_ref}
    headers = signed_subnet_headers(wallet, '/submit')

    status, response = _post_json(f"{cfg.subnet_api_url.rstrip('/')}/submit", body, headers)
    if 200 <= status < 300:
        click.echo(f"Success: {response}")
        return

    click.echo(f"Error ({status}): {response}")
    sys.exit(1)


if __name__ == "__main__":
    main()
