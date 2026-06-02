"""Tracking JSONL file lifecycle management."""

import datetime
import json
import os
import time
from pathlib import Path
from typing import Any, TextIO


class TrackingJsonlWriter:
    """Owns tracking JSONL paths, rotation, durability, and frame numbering."""

    def __init__(
        self,
        directory: str | Path,
        *,
        fsync_interval_s: float = 1.0,
    ) -> None:
        self.directory = Path(directory)
        self.fsync_interval_s = fsync_interval_s
        self.path: str | None = None
        self.recording_path: str | None = None
        self.last_completed_path: str | None = None
        self.frame_count = 0
        self._file: TextIO | None = None
        self._last_fsync_ts = 0.0

    @property
    def is_open(self) -> bool:
        return self._file is not None

    def make_path(self, ts: str | None = None, label: str | None = None) -> str:
        """Return a unique tracking_*.jsonl path under the configured directory."""
        if ts is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"tracking_{ts}"
        if label:
            base += f"_{label}"
        candidate = self.directory / f"{base}.jsonl"
        idx = 1
        while candidate.exists():
            candidate = self.directory / f"{base}_{idx}.jsonl"
            idx += 1
        return str(candidate)

    def open(self, path: str, *, reset_counter: bool) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "w", encoding="utf-8", buffering=1)
        self.path = path
        if reset_counter:
            self.frame_count = 0
        self._last_fsync_ts = time.time()

    def rotate(self, path: str, *, reset_counter: bool) -> None:
        self.close(force_fsync=True)
        self.open(path, reset_counter=reset_counter)

    def append(self, row: dict[str, Any], *, bump_frame_counter: bool) -> None:
        if self._file is None:
            return
        if bump_frame_counter:
            row = {"frame": self.frame_count, **row}
        try:
            self._file.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.flush(force_fsync=False)
            if bump_frame_counter:
                self.frame_count += 1
        except Exception:
            pass

    def flush(self, *, force_fsync: bool = False) -> None:
        if self._file is None:
            return
        try:
            self._file.flush()
        except Exception:
            return
        now = time.time()
        if force_fsync or now - self._last_fsync_ts >= self.fsync_interval_s:
            try:
                os.fsync(self._file.fileno())
                self._last_fsync_ts = now
            except Exception:
                pass

    def close(self, *, force_fsync: bool = True) -> None:
        if self._file is None:
            return
        self.flush(force_fsync=force_fsync)
        try:
            self._file.close()
        except Exception:
            pass
        self._file = None

    def latest_path(self, *, recording: bool) -> str | None:
        if recording and self.path:
            return self.path
        if self.last_completed_path and Path(self.last_completed_path).exists():
            return self.last_completed_path
        if self.path:
            return self.path
        jsonls = sorted(self.directory.glob("tracking_*.jsonl"), reverse=True)
        return str(jsonls[0]) if jsonls else None
