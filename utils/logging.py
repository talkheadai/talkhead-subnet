import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone

import bittensor as bt
import wandb


WANDB_MAX_LOGS = 95_000
WANDB_MAX_RUN_AGE_DAYS = 1
EVENTS_LEVEL_NUM = 38
DEFAULT_LOG_BACKUP_COUNT = 10

def maybe_reset_wandb(validator):
    if validator.config.wandb.off or wandb.run is None:
        return

    today = datetime.now(timezone.utc).date()

    # Prefer the local cached start date, fall back to wandb metadata, finally today's date.
    run_start_date = getattr(validator, "_wandb_start_date", None)
    if run_start_date is None:
        start_time = getattr(wandb.run, "start_time", None)
        if isinstance(start_time, datetime):
            run_start_date = start_time.date()
        elif isinstance(start_time, str):
            try:
                run_start_date = datetime.fromisoformat(start_time).date()
            except ValueError:
                run_start_date = None

    if run_start_date is None:
        run_start_date = today
    validator._wandb_start_date = run_start_date

    if (today - run_start_date).days >= WANDB_MAX_RUN_AGE_DAYS:
        bt.logging.info("Restarting W&B to rotate runs daily.")
        wandb.run.finish()
        validator.init_wandb()
        return

    log_file_path = os.path.join(wandb.run.dir, "output.log")
    try:
        with open(log_file_path, "rb") as f:
            num_lines = sum(1 for _ in f)
    except FileNotFoundError:
        return
    except OSError as e:
        bt.logging.debug(f"Skipping W&B log rotation check: {e}")
        return

    if num_lines > WANDB_MAX_LOGS:
        bt.logging.info("Restarting W&B to limit the number of logs per run!...")
        wandb.run.finish()
        validator.init_wandb()
