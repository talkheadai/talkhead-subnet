import os
import subprocess
import sys
import configparser
import bittensor as bt
import wandb


def run_git_command(command, check=True, capture_output=False):
    try:
        result = subprocess.run(
            command,
            capture_output=capture_output,
            text=True,
            check=check,
        )
        return result.stdout.strip() if capture_output else None
    except subprocess.CalledProcessError as e:
        bt.logging.error(f"Git command failed: {' '.join(command)}\n{e}")
        return None


def get_current_branch():
    return run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True)


def has_uncommitted_changes():
    return bool(run_git_command(["git", "status", "--porcelain"], capture_output=True))


def is_up_to_date_with_main():

    bt.logging.info("Fetching latest commits from origin/main")
    run_git_command(["git", "fetch", "origin", "main"], check=True)

    main_commit = run_git_command(["git", "rev-parse", "origin/main"], capture_output=True)
    current_commit = run_git_command(["git", "rev-parse", "HEAD"], capture_output=True)

    if main_commit and current_commit and main_commit == current_commit:
        bt.logging.info("Current branch is up to date with origin/main")
        return True
    bt.logging.error("Current branch is not up to date with origin/main")
    return False


def update_to_latest():

    current_branch = get_current_branch()
    if not current_branch:
        bt.logging.error("Failed to determine current git branch. Is this a git repository?")
        sys.exit(1)

    bt.logging.info(f"Current git branch: {current_branch}")

    if is_up_to_date_with_main():
        bt.logging.info("Repository is already up to date with the latest code from main")
        return True

    bt.logging.info("Repository is not up to date. Starting update process...")

    if current_branch != "main":
        bt.logging.info(f"Switching from '{current_branch}' to 'main' branch...")
        run_git_command(["git", "checkout", "main"])

    bt.logging.info("Pulling latest changes from origin/main...")
    run_git_command(["git", "pull", "origin", "main"])

    if is_up_to_date_with_main():
        bt.logging.info("Successfully updated to latest code from main")
        # Cleanly finish any running WandB session before restarting the process.
        try:
            if wandb.run is not None:
                bt.logging.info("Finishing active W&B run before restart.")
                wandb.run.finish()
        except Exception as e:
            bt.logging.debug(f"Failed to finish W&B run before restart: {e}")
        python = sys.executable
        # restart the project.
        bt.logging.info("Restart the project.")
        os.execv(python, [python] + sys.argv)
    bt.logging.error("Failed to update repository. Exiting...")
    sys.exit(1)


def check_and_update_code():
    bt.logging.info("Starting automatic code update process...")
    return update_to_latest()
