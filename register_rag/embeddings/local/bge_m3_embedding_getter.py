from typing import List

from sentence_transformers import SentenceTransformer
from torch import device

from ...entity import Document
from .local_embedding_getter import LocalEmbeddingGetter


class BgeM3EmbeddingGetter(LocalEmbeddingGetter):
    def __init__(
        self,
        weight_path: str,
        pre_load: bool = False,
        device: device = device("cuda"),
    ) -> None:
        """Initialize the BgeM3EmbeddingGetter class

        Args:
            weight_path (str): the path to the model weights

            pre_load (bool, optional):  If True, the model will be loaded
            when the object is created. If False, the model will be loaded
            when the `get_embedding` method is called. Defaults to False.

            device (device, optional): The device to use for the model.
            Defaults to device("cuda").
        """
        super().__init__(weight_path, pre_load)

        self.device = device

    async def load(self):
        self.model = SentenceTransformer(
            model_name_or_path=self.weight_path, device=self.device
        )

    async def embedding(self, docs: List[Document]) -> List[List[float]]:
        return self.model.encode(
            sentences=[doc.page_content for doc in docs], device=self.device
        )
