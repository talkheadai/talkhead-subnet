"""TalkHead SN108 axon protocol definitions."""

from typing import Optional

import bittensor as bt
from pydantic import Field


class ImageRef(bt.Synapse):
    """
    Decentralized image reference registration synapse for TalkHead subnet (SN108).

    Validators query a miner's axon with an empty ``ImageRef`` request.
    The miner responds by populating ``image_ref``.
    each miner advertises its current container image directly on its axon.

    The ``image_ref`` must be an immutable OCI reference including a digest,
    e.g. ``talkheadai/talkhead-miner@sha256:abc123...``. Validators and the
    executor use this value to pull and evaluate the miner's image container.

    Attributes:
        image_ref: OCI image reference with ``@sha256:`` digest, set by the miner.
    """

    image_ref: Optional[str] = Field(
        default=None,
        title="Image reference",
        description=(
            "OCI image reference pinned by digest, e.g. "
            "'talkheadai/talkhead-miner@sha256:abc123...'"
        ),
        examples=["talkheadai/talkhead-miner@sha256:abc123..."],
    )

    def deserialize(self) -> str:
        """
        Return image reference after a dendrite query.

        Returns:
            Image reference
            for validator-side processing.
        """
        return self.image_ref
