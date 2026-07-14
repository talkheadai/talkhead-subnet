from __future__ import annotations

import os

from loguru import logger


def resolve_hf_token() -> str | None:
    """
    Return a Hugging Face Hub token from the environment, if configured.

    Accepts either HF_TOKEN (preferred) or HUGGING_FACE_HUB_TOKEN.
    """
    token = (os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or "").strip()
    return token or None


def configure_hf_auth() -> str | None:
    """
    Ensure Hugging Face downloads can use an authenticated session.

    Propagates HF_TOKEN into the process env when present so huggingface_hub /
    datasets pick it up, and logs a short status without revealing the secret.
    """
    token = resolve_hf_token()
    if token:
        os.environ["HF_TOKEN"] = token
        logger.info("Hugging Face auth: HF_TOKEN configured")
        return token

    logger.warning(
        "Hugging Face auth: no HF_TOKEN set; downloads may hit lower rate limits. "
        "Set HF_TOKEN in your environment or .env file."
    )
    return None
