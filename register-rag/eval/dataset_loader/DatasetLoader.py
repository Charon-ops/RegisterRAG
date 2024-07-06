from abc import ABC, abstractmethod

from configs.DatasetConfig import DatasetConfig


class DatasetLoader(ABC):
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        self.config = self.load_config()

    @abstractmethod
    def load_config(self) -> DatasetConfig:
        raise NotImplementedError("load_data method must be implemented in a subclass")
