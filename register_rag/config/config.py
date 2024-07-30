import os
import json
from pydantic import BaseModel

from .embedding_config import EmbeddingConfig
from .store_config import StoreConfig
from .generation_config import GenerationConfig
from .prompt_config import PromptConfig


class Config(BaseModel):
    """
    The configuration class for the RAG model.

    The configuration class contains the following attributes:
    - embedding: EmbeddingConfig
    - store: StoreConfig
    - generation: GenerationConfig
    - prompt: PromptConfig

    PromptConfig is optional and has default values. The default template is:

    ```plaintext
    请根据下面的信息：
    {recall_res}
    回答问题：{query}
    ```

    where {recall_res} is the retrieved document content and {query} is the query.
    """

    embedding: EmbeddingConfig
    store: StoreConfig
    generation: GenerationConfig
    prompt: PromptConfig = PromptConfig()

    @classmethod
    def load(cls, path: str) -> "Config":
        if not os.path.isfile(path):
            raise ValueError(f"Config file not found at {path}")
        if not path.endswith(".json"):
            raise ValueError("Config file must be a JSON file")
        config = json.load(open(path))
        return cls(**config)
