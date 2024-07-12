import asyncio

from transformers import AutoModel

from ..embedding_getter import EmbeddingGetter


class LocalEmbeddingGetter(EmbeddingGetter):
    """
    Class: LocalEmbeddingGetter, A subclass of `EmbeddingGetter`

    Purpose:
        Retrieves embeddings for documents using a local model. This class serves as a flexible
        base for embedding extraction with customizable pre- and post-processing steps.
    """

    def __init__(self, weight_path: str, pre_load: bool = False) -> None:
        super().__init__()
        self.weight_path = weight_path
        self.model = None
        if pre_load:
            self.load_task = asyncio.create_task(self.load())

    async def load(self):
        self.model = AutoModel.from_pretrained(self.weight_path)
