import asyncio
from abc import abstractmethod

from ..embedding_getter import EmbeddingGetter


class LocalEmbeddingGetter(EmbeddingGetter):
    """A class to get the embeddings of the documents using the local embeddings.

    If you want to use a different local embedding getter, you can just inherit the EmbeddingGetter class and implement the get_embedding method.
    You should implement the load method to load the embeddings.

    The load method will be auto called when the class is initialized.
    """

    def __init__(self, weight_path: str) -> None:
        super().__init__()
        self.weight_path = weight_path
        self.load_task = asyncio.create_task(self.load())

    async def load(self):
        raise NotImplementedError(
            "load method is not implemented, it should be implemented in the sub class."
        )
