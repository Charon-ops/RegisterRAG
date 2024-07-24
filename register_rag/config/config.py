import os
import json
from pydantic import BaseModel

from .embedding_config import EmbeddingConfig
from .store_config import StoreConfig
from .generation_config import GenerationConfig


class Config(BaseModel):
    embedding: EmbeddingConfig
    store: StoreConfig
    generation: GenerationConfig

    @classmethod
    def load(cls, path: str) -> "Config":
        if not os.path.isfile(path):
            raise ValueError(f"Config file not found at {path}")
        if not path.endswith(".json"):
            raise ValueError("Config file must be a JSON file")
        config = json.load(open(path))
        return cls(**config)
