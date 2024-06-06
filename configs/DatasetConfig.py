from typing import Dict, Any
import os
import json

from .base import Config


class DatasetConfig(Config):
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        self._config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        config_path = os.path.join(self.data_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
