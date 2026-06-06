"""Experiment logging and result tracking."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

class ExperimentLogger:
    def __init__(self, exp_dir: str | Path, exp_name: str = ""):
        self.exp_dir = Path(exp_dir)
        if exp_name:
            self.exp_dir = self.exp_dir / exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.exp_dir / "log.json"
        self.entries: list[dict] = []
        self._start_time = time.time()

    def log(self, step: int | None = None, **kwargs: Any) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_s": round(time.time() - self._start_time, 2),
        }
        if step is not None:
            entry["step"] = step
        entry.update(kwargs)
        self.entries.append(entry)
        self._flush()

    def metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.log(step=step, **metrics)

    def finalize(self, summary: dict[str, Any] | None = None) -> str:
        data = {
            "total_entries": len(self.entries),
            "total_elapsed_s": round(time.time() - self._start_time, 2),
        }
        if summary:
            data["summary"] = summary
        self.log(**data)
        return str(self.log_path)

    def _flush(self) -> None:
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=2, ensure_ascii=False)
