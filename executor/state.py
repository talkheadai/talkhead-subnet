from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from loguru import logger

from executor.models import MinerMetricsResponse, MinerRecord, MinerSubmission
from talkhead.constant import RESUBMIT_COOLDOWN_SEC

ROUND_CONTINUE_TOP_K = 10
_TERMINAL_ERRORS = frozenset({"Blacklisted", "Docker pull failed"})
_DROPPED_FINAL_SCORE = 0.0


def _metrics_error(metrics: dict[str, Any] | None) -> str | None:
    if not isinstance(metrics, dict):
        return None
    raw_err = metrics.get("error")
    return raw_err if isinstance(raw_err, str) else None


def _metrics_final_score(metrics: dict[str, Any] | None) -> float | None:
    if not isinstance(metrics, dict):
        return None
    raw_score = metrics.get("final_score")
    if isinstance(raw_score, (int, float)):
        return float(raw_score)
    return None


def _mark_dropped_from_competition(
    metrics: dict[str, Any],
    *,
    top_final_score: float,
) -> dict[str, Any]:
    score = _metrics_final_score(metrics)
    if score is None:
        return metrics
    updated = dict(metrics)
    updated["dropped_from_competition"] = True
    updated["final_score_diff_from_top"] = top_final_score - score
    updated["round_final_score"] = score
    updated["final_score"] = _DROPPED_FINAL_SCORE
    return updated


def _resolve_db_and_legacy_paths(state_file: Path) -> tuple[Path, Path | None]:
    if state_file.suffix.lower() == ".json":
        return state_file.with_suffix(".db"), state_file
    return state_file, state_file.with_suffix(".json")


class MinerState:
    def __init__(self, state_file: str | None = None) -> None:
        self._lock = threading.Lock()
        self._miners: dict[str, MinerRecord] = {}
        if state_file:
            raw = Path(state_file)
            self._db_path, self._legacy_json_path = _resolve_db_and_legacy_paths(raw)
        else:
            self._db_path = None
            self._legacy_json_path = None
        self._load_from_disk()

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=WAL")
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(miners)").fetchall()
        }
        if columns and "submit_time" in columns and "updated_time" not in columns:
            conn.execute(
                "ALTER TABLE miners RENAME COLUMN submit_time TO updated_time"
            )
            columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(miners)").fetchall()
            }
        if columns and "updated_time" not in columns:
            conn.execute(
                "ALTER TABLE miners ADD COLUMN updated_time REAL NOT NULL DEFAULT 0"
            )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS miners (
                hotkey TEXT PRIMARY KEY NOT NULL,
                image_ref TEXT NOT NULL,
                updated_time REAL NOT NULL,
                metrics_json TEXT,
                coming_metrics_json TEXT
            )
            """
        )

    @staticmethod
    def _miners_from_json_payload(payload: object) -> dict[str, MinerRecord]:
        if not isinstance(payload, dict):
            return {}
        miners = payload.get("miners")
        if not isinstance(miners, list):
            return {}

        loaded: dict[str, MinerRecord] = {}
        for item in miners:
            if not isinstance(item, dict):
                continue
            hotkey = item.get("hotkey")
            image_ref = item.get("image_ref")
            metrics = item.get("metrics")
            coming_metrics = item.get("coming_metrics")
            updated_time = item.get("updated_time", item.get("submit_time", 0.0))
            if not isinstance(hotkey, str) or not isinstance(image_ref, str):
                continue
            if not isinstance(updated_time, (int, float)):
                updated_time = 0.0
            loaded[hotkey] = MinerRecord(
                hotkey=hotkey,
                image_ref=image_ref,
                updated_time=float(updated_time),
                metrics=metrics if isinstance(metrics, dict) else None,
                coming_metrics=coming_metrics if isinstance(coming_metrics, dict) else None,
            )
        return loaded

    def _load_miners_from_json_file(self, path: Path) -> dict[str, MinerRecord]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"failed to load legacy state file {path}: {exc}")
            return {}
        return self._miners_from_json_payload(payload)

    def _load_from_disk(self) -> None:
        if self._db_path is None:
            return

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        db_existed = self._db_path.exists()

        conn = sqlite3.connect(str(self._db_path))
        try:
            self._ensure_schema(conn)
            if db_existed:
                cur = conn.execute(
                    "SELECT hotkey, image_ref, updated_time, metrics_json, coming_metrics_json "
                    "FROM miners ORDER BY hotkey"
                )
                for (
                    hotkey,
                    image_ref,
                    updated_time,
                    metrics_json,
                    coming_metrics_json,
                ) in cur.fetchall():
                    metrics: dict[str, Any] | None = None
                    coming_metrics: dict[str, Any] | None = None
                    if isinstance(metrics_json, str) and metrics_json.strip():
                        try:
                            parsed = json.loads(metrics_json)
                            if isinstance(parsed, dict):
                                metrics = parsed
                        except Exception:
                            metrics = None
                    if isinstance(coming_metrics_json, str) and coming_metrics_json.strip():
                        try:
                            parsed = json.loads(coming_metrics_json)
                            if isinstance(parsed, dict):
                                coming_metrics = parsed
                        except Exception:
                            coming_metrics = None
                    self._miners[hotkey] = MinerRecord(
                        hotkey=hotkey,
                        image_ref=image_ref,
                        updated_time=float(updated_time),
                        metrics=metrics,
                        coming_metrics=coming_metrics,
                    )
                logger.info(f"loaded {len(self._miners)} miners from sqlite")
        finally:
            conn.close()

        if not db_existed and self._legacy_json_path and self._legacy_json_path.exists():
            loaded = self._load_miners_from_json_file(self._legacy_json_path)
            if loaded:
                with self._lock:
                    self._miners = loaded
                    self._save_locked()
                logger.info(
                    f"migrated {len(loaded)} miners from {self._legacy_json_path} to sqlite"
                )

    def _save_locked(self) -> None:
        if self._db_path is None:
            return

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            (
                r.hotkey,
                r.image_ref,
                r.updated_time,
                json.dumps(r.metrics) if isinstance(r.metrics, dict) else None,
                json.dumps(r.coming_metrics) if isinstance(r.coming_metrics, dict) else None,
            )
            for r in sorted(self._miners.values(), key=lambda r: r.hotkey)
        ]
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._ensure_schema(conn)
            conn.execute("DELETE FROM miners")
            conn.executemany(
                "INSERT INTO miners (hotkey, image_ref, updated_time, metrics_json, coming_metrics_json) "
                "VALUES (?,?,?,?,?)",
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    def upsert_submissions(self, submissions: list[MinerSubmission]) -> None:
        changed = False
        now = time.time()
        with self._lock:
            for submission in submissions:
                existing = self._miners.get(submission.hotkey)
                if existing is None:
                    self._miners[submission.hotkey] = MinerRecord(
                        hotkey=submission.hotkey,
                        image_ref=submission.image_ref,
                        updated_time=now,
                    )
                    changed = True
                    logger.info(f"new miner added: hotkey={submission.hotkey}")
                    continue

                if existing.image_ref == submission.image_ref:
                    continue

                elapsed = now - existing.updated_time
                if elapsed < RESUBMIT_COOLDOWN_SEC:
                    logger.info(
                        f"image_ref update skipped (cooldown): hotkey={submission.hotkey} "
                        f"elapsed_sec={elapsed:.0f} cooldown_sec={RESUBMIT_COOLDOWN_SEC}"
                    )
                    continue

                self._miners[submission.hotkey] = MinerRecord(
                    hotkey=submission.hotkey,
                    image_ref=submission.image_ref,
                    updated_time=now,
                    metrics=None,
                    coming_metrics=None,
                )
                changed = True
                logger.info(
                    f"image_ref updated: hotkey={submission.hotkey} image_ref={submission.image_ref}"
                )
            if changed:
                self._save_locked()

    def list_metrics(self) -> list[MinerMetricsResponse]:
        with self._lock:
            records = sorted(self._miners.values(), key=lambda r: r.hotkey)
            return [
                MinerMetricsResponse(
                    hotkey=record.hotkey,
                    image_ref=record.image_ref,
                    updated_time=record.updated_time,
                    metrics=record.metrics,
                )
                for record in records
                if record.metrics is not None
            ]

    def has_miners(self) -> bool:
        with self._lock:
            return bool(self._miners)

    def get_pending_miners(self) -> list[MinerRecord]:
        with self._lock:
            pending = [m for m in self._miners.values() if m.coming_metrics is None]
            pending.sort(key=lambda r: r.hotkey)
            return [replace(record) for record in pending]

    def set_evaluation_result(
        self,
        *,
        hotkey: str,
        image_ref: str,
        metrics: dict[str, Any] | None,
    ) -> bool:
        with self._lock:
            record = self._miners.get(hotkey)
            if record is None:
                return False
            if record.image_ref != image_ref:
                return False
            record.coming_metrics = metrics
            self._save_locked()
            return True

    def commit_round_if_complete(self) -> bool:
        with self._lock:
            if not self._miners:
                return False
            if any(record.coming_metrics is None for record in self._miners.values()):
                return False

            topk_candidates: list[tuple[float, str]] = []
            for record in self._miners.values():
                next_metrics = record.coming_metrics
                if not isinstance(next_metrics, dict):
                    record.metrics = next_metrics
                    record.coming_metrics = next_metrics
                    continue

                record.metrics = next_metrics
                err = _metrics_error(next_metrics)
                if err in _TERMINAL_ERRORS:
                    record.coming_metrics = next_metrics
                    continue

                final_score = _metrics_final_score(next_metrics)
                if final_score is not None:
                    topk_candidates.append((final_score, record.hotkey))
                record.coming_metrics = next_metrics

            topk_candidates.sort(key=lambda item: (-item[0], item[1]))
            continue_hotkeys = {
                hotkey for _, hotkey in topk_candidates[:ROUND_CONTINUE_TOP_K]
            }
            top_final_score = topk_candidates[0][0] if topk_candidates else None

            for record in self._miners.values():
                if record.hotkey in continue_hotkeys:
                    record.coming_metrics = None
                    continue

                metrics = record.metrics
                if not isinstance(metrics, dict):
                    continue
                if _metrics_error(metrics) in _TERMINAL_ERRORS:
                    continue

                round_score = _metrics_final_score(metrics)
                if round_score is None:
                    continue

                if top_final_score is None:
                    continue

                dropped = _mark_dropped_from_competition(
                    metrics,
                    top_final_score=top_final_score,
                )
                record.metrics = dropped
                record.coming_metrics = dropped

            self._save_locked()
            return True
