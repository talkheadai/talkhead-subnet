"""
Deprecated standalone HTTP executor.

The executor now runs in-process inside ``neurons.validator``.
Use ``python -m neurons.validator`` instead of starting this module.
"""

from __future__ import annotations

import warnings


def run() -> None:
    warnings.warn(
        "executor.app is deprecated; run `python -m neurons.validator` instead",
        DeprecationWarning,
        stacklevel=2,
    )
    raise SystemExit(
        "executor.app is deprecated. The executor runs inside neurons.validator."
    )


if __name__ == "__main__":
    run()
