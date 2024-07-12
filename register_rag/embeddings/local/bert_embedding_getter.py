from typing import List

from transformers import BertModel, AutoTokenizer
from torch import no_grad

from ...entity import Document
from .local_embedding_getter import LocalEmbeddingGetter


class BertEmbeddingGetter(LocalEmbeddingGetter):
    def __init__(self, weight_path: str, pre_load: bool = False) -> None:
        super().__init__(weight_path, pre_load)
        self.tokenizer = None

    async def load(self):
        self.model = BertModel.from_pretrained(
            pretrained_model_name_or_path=self.weight_path
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.weight_path
        )

    async def embedding(self, docs: List[Document]) -> List[List[float]]:
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
