"""State management for tracking incremental model runs"""

import json
from pathlib import Path
from typing import Any, Dict


class StateManager:
    """Manages state for incremental models"""

    def __init__(self, state_file: Path):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state: Dict[str, Dict[str, Any]] = self._load_state()

    def _load_state(self) -> Dict[str, Dict[str, Any]]:
        """Load state from JSON file"""
        if not self.state_file.exists():
            return {}

        with open(self.state_file, 'r') as f:
            return json.load(f)

    def _save_state(self):
        """Save state to JSON file"""
        with open(self.state_file, 'w') as f:
            json.dump(self._state, f, indent=2)

    def get_state(self, model_name: str) -> Dict[str, Any]:
        """Get state for a specific model"""
        return self._state.get(model_name, {})

    def update_state(self, model_name: str, state: Dict[str, Any]):
        """Update state for a specific model"""
        if model_name not in self._state:
            self._state[model_name] = {}

        self._state[model_name].update(state)
        self._save_state()

    def clear_state(self, model_name: str):
        """Clear state for a specific model (forces full refresh)"""
        if model_name in self._state:
            del self._state[model_name]
            self._save_state()

    def clear_all(self):
        """Clear all state"""
        self._state = {}
        self._save_state()
