from typing import Dict, Any
import os
import json

from .DatasetConfig import DatasetConfig


class OurDatasetConfig(DatasetConfig):
    def __init__(self, data_path: str) -> None:
        super().__init__(data_path)

    def load_config(self) -> Dict[str, Any]:
        config_path = os.path.join(self.data_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        assert "insert" in config, "config does not have enough keys"
        if config["insert"]:
            assert self.config.keys() >= {
                "chunk_size",
                "embedding_router_path",
                "embedding_port",
                "store_router_path",
                "store_port",
                "store_name",
            }, "config does not have enough keys"
        else:
            assert self.config.keys() >= {
                "store_name",
            }, "config does not have enough keys"

        return config

    @property
    def insert(self) -> bool:
        if "insert" not in self._config:
            raise ValueError("insert not in config, please confirm your config file.")
        return self._config["insert"]

    @property
    def chunk_size(self) -> int:
        if "chunk_size" not in self._config:
            raise ValueError("chunk_size not in config, maybe you don't need it?")
        return self._config["chunk_size"]

    @property
    def embedding_router_path(self) -> str:
        if "embedding_router_path" not in self._config:
            raise ValueError(
                "embedding_router_path not in config, maybe you don't need it?"
            )
        return self._config["embedding_router_path"]

    @property
    def embedding_port(self) -> int:
        if "embedding_port" not in self._config:
            raise ValueError("embedding_port not in config, maybe you don't need it?")
        return self._config["embedding_port"]

    @property
    def store_router_path(self) -> str:
        if "store_router_path" not in self._config:
            raise ValueError(
                "store_router_path not in config, maybe you don't need it?"
            )
        return self._config["store_router_path"]

    @property
    def store_port(self) -> int:
        if "store_port" not in self._config:
            raise ValueError("store_port not in config, maybe you don't need it?")
        return self._config["store_port"]

    @property
    def store_name(self) -> str:
        if "store_name" not in self._config:
            raise ValueError(
                "store_name not in config, please confirm your config file."
            )
        return self._config["store_name"]
