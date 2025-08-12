# session_memory.py

from typing import Dict, Any, List
import datetime
import json
import os


class SessionMemory:
    """
    A lightweight session memory manager for the autonomous research assistant.
    Stores and retrieves conversation history and context across user sessions.
    """

    def __init__(self, session_id: str, storage_path: str = "./memory_store"):
        self.session_id = session_id
        self.storage_path = storage_path
        self.file_path = os.path.join(storage_path, f"{session_id}.json")

        os.makedirs(storage_path, exist_ok=True)
        self.memory = self._load_memory()

    def _load_memory(self) -> Dict[str, Any]:
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {"history": [], "metadata": {}}
        return {"history": [], "metadata": {}}

    def _save_memory(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, indent=2)

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.

        Args:
            role (str): 'user', 'assistant', or 'system'.
            content (str): The message content.
        """
        self.memory["history"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.utcnow().isoformat()
        })
        self._save_memory()

    def get_history(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history.

        Args:
            limit (int): Optional limit on number of messages.
        """
        if limit is None:
            return self.memory["history"]
        return self.memory["history"][-limit:]

    def set_metadata(self, key: str, value: Any) -> None:
        """
        Store metadata for the session (e.g., research topic, search queries).
        """
        self.memory["metadata"][key] = value
        self._save_memory()

    def get_metadata(self, key: str, default=None) -> Any:
        """
        Retrieve metadata for the session.
        """
        return self.memory["metadata"].get(key, default)

    def clear(self) -> None:
        """
        Clears all session memory for this ID.
        """
        self.memory = {"history": [], "metadata": {}}
        self._save_memory()


# Example usage:
if __name__ == "__main__":
    mem = SessionMemory("test_session")
    mem.add_message("user", "What's the latest research on quantum computing?")
    mem.add_message("assistant", "Here's a 2025 paper from Nature...")
    mem.set_metadata("topic", "quantum computing")
    print(mem.get_history())
    print(mem.get_metadata("topic"))
