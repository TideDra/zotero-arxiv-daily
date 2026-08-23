from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from .protocol import Paper


@dataclass
class AdsState:
    seen: dict[str, str] = field(default_factory=dict)
    version: int = 1

    @classmethod
    def load(cls, path: str | Path) -> "AdsState":
        state_path = Path(path)
        if not state_path.exists():
            return cls()
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid SciX/ADS-compatible state file: {state_path}") from exc
        if payload.get("version") != 1 or not isinstance(payload.get("seen"), dict):
            raise ValueError(f"Unsupported SciX/ADS-compatible state schema: {state_path}")
        return cls(seen={str(key): str(value) for key, value in payload["seen"].items()})

    def filter_new(self, papers: list[Paper]) -> list[Paper]:
        return [paper for paper in papers if paper.stable_id not in self.seen]

    def mark_seen(self, papers: list[Paper]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        for paper in papers:
            timestamp = paper.entry_at.isoformat() if paper.entry_at else now
            self.seen[paper.stable_id] = timestamp

    def prune(self, retention_days: int = 90) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        retained: dict[str, str] = {}
        for stable_id, value in self.seen.items():
            try:
                timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if timestamp >= cutoff:
                retained[stable_id] = value
        self.seen = retained

    def save(self, path: str | Path) -> None:
        state_path = Path(path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "seen": dict(sorted(self.seen.items())),
        }
        temporary_path = state_path.with_suffix(state_path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary_path.replace(state_path)
