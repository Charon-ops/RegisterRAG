from abc import ABC, abstractmethod
from typing import Dict, Any, List
import os
import json

from entity.evaluate_params import EvaluateConfig


class Evaluator(ABC):
    def __init__(self, config_path: str) -> None:
        assert os.path.exists(
            config_path
        ), f"config_file: {config_path} does not exist."
        self.config_path = config_path
        self.config = self.load_config()
        self.query: List[str] = None
        self.response: List[str] = None

    def load_config(self) -> EvaluateConfig:
        with open(self.config_path, "r") as f:
            config = json.load(f)
        config = self.init_config(config)
        return config

    def init_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return config

    @abstractmethod
    def evaluate(self, save_path: str) -> None:
        return NotImplementedError("evaluate method must be implemented in a subclass.")
