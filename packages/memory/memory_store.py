import json
import os
from pathlib import Path
from datetime import datetime

DEFAULT_MEMORY_PATH = Path("data/memory.json")

class MemoryStore:
    def __init__(self, file_path: str = None):
        self.file_path = Path(file_path) if file_path else DEFAULT_MEMORY_PATH
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        if not self.file_path.exists():
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump({"sessions": []}, f, indent=2)

    def _load(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def add_session(self, goal: str, results: dict):
        """
        Store a completed research session.

        Args:
            goal (str): Research goal or query.
            results (dict): Summaries, insights, and sources.
        """
        data = self._load()
        session = {
            "timestamp": datetime.utcnow().isoformat(),
            "goal": goal,
            "results": results
        }
        data["sessions"].append(session)
        self._save(data)

    def list_sessions(self):
        """Return a list of stored research sessions."""
        return self._load()["sessions"]

    def get_last_session(self):
        """Return the most recent research session."""
        sessions = self.list_sessions()
        return sessions[-1] if sessions else None
