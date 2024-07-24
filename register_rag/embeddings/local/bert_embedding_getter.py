from typing import List

from transformers import BertModel, AutoTokenizer
from torch import no_grad

from ...config import Config
from ...documents import Document
from .local_embedding_getter import LocalEmbeddingGetter


class BertEmbeddingGetter(LocalEmbeddingGetter):
    """
    Class: BertEmbeddingGetter

    Purpose:
        Retrieves embeddings for documents using the Bert model.

    Note:
        This class treats the `[CLS]` token as the document embedding.
    """

    def __init__(self, config: Config) -> None:
        """
        Initialize the BertEmbeddingGetter

        Args:
            weight_path (str): the path to the model weights, or you can also use
            the model name from huggingface.

            pre_load (bool, optional): If True, the model will be loaded
            when the object is created. If False, the model will be loaded
            when the `get_embedding` method is called. Defaults to False.
        """
        super().__init__(config)
        self.tokenizer = None

    async def load(self):
        """
        Load the model. The model and tokenizer are loaded from the weight path.
        """
        self.model = BertModel.from_pretrained(
            pretrained_model_name_or_path=self.weight_path
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.weight_path
        )

    async def embedding(self, docs: List[Document]) -> List[List[float]]:
        """
        Get the embeddings of the documents using the Bert model.
        The `[CLS]` token is used as the document embedding.

        Args:
            docs (List[Document]): A list of documents to get the embeddings.

        Returns:
            List[List[float]]: A list of embeddings of the documents.
        """
        encoded_input = self.tokenizer(
            [doc.page_content for doc in docs],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with no_grad():
            model_output = self.model(**encoded_input)
        res = model_output.pooler_output.tolist()
        return res
