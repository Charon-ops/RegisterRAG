import asyncio

from transformers import AutoModel

from ...config import Config
from ..embedding_getter import EmbeddingGetter


class LocalEmbeddingGetter(EmbeddingGetter):
    """
    Class: LocalEmbeddingGetter, A subclass of `EmbeddingGetter`

    Purpose:
        Retrieves embeddings for documents using a local model. This class serves as a flexible
        base for embedding extraction with customizable pre- and post-processing steps.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if self.config.embedding.embedding_model_name_or_path is None:
            raise ValueError("Model name or path is required.")

        self.weight_path = self.config.embedding.embedding_model_name_or_path
        self.device = self.config.embedding.embedding_model_device
        self.preload = self.config.embedding.embedding_model_preload
        self.model = None
        if self.preload:
            self.load_task = asyncio.create_task(self.load())

    async def load(self):
        self.model = AutoModel.from_pretrained(self.weight_path)
