from typing import List
import os

from sentence_transformers import SentenceTransformer
from torch import device

from ...config import Config
from ...documents import Document
from ...exceptions.embedding_exceptions import WeightPathNotValidException

from .local_embedding_getter import LocalEmbeddingGetter


class SentenceTransformerEmbeddingGetter(LocalEmbeddingGetter):
    """
    Class: BgeM3EmbeddingGetter

    Purpose:
        Retrieves embeddings for documents using the Bge-m3 model.
    """

    def __init__(self, config: Config) -> None:
        """
        Initialize the BgeM3EmbeddingGetter class

        Args:
            weight_path (str): the path to the model weights, or you can also use
            the model name from huggingface.

            pre_load (bool, optional):  If True, the model will be loaded
            when the object is created. If False, the model will be loaded
            when the `get_embedding` method is called. Defaults to False.

            device (device, optional): The device to use for the model.
            Defaults to device("cuda").
        """
        super().__init__(config)
        self.weight_path = "/".join(self.weight_path.split("/")[1:])

    async def load(self):
        """
        Load the model.

        Raises:
            WeightPathNotValidException: If the weight path is not a directory.
        """
        if not os.path.isdir(self.weight_path):
            if "/" not in self.weight_path:
                raise WeightPathNotValidException(self.__class__.__name__)
        self.model = SentenceTransformer(
            model_name_or_path=self.weight_path, device=self.device
        )

    async def embedding(self, docs: List[Document]) -> List[List[float]]:
        """
        Embedding method for Bge-m3.

        Args:
            docs (List[Document]): A list of documents to embed.

        Returns:
            List[List[float]]: A list of embeddings for each document.
        """
        return self.model.encode(
            sentences=[doc.page_content for doc in docs], device=self.device
        )
