from typing import List, Dict
import os
import json

from langchain_core.documents import Document

from splitter import RecursiveSplitter
from embedding import BgeEmbedding
from store import AnnStore
from config import OurDatasetConfig
from .DatasetLoader import DatasetLoader


class OurDatasetLoader(DatasetLoader):
    def __init__(self, data_path: str) -> None:
        super().__init__(data_path)

    def load_config(self) -> OurDatasetConfig:
        return OurDatasetConfig(self.data_path)

    def insert_data(self) -> None:
        data_path = os.path.join(self.data_path, "data.txt")
        with open(data_path, "r") as f:
            data = f.read()

        splitter = RecursiveSplitter(chunk_size=self.config.chunk_size)
        chunks = splitter.split(Document(page_content=data))

        embedding = BgeEmbedding(
            router_path=self.config.embedding_router_path,
            port=self.config.embedding_port,
        )
        embds = embedding.embed_documents(chunks)

        store = AnnStore(
            router_path=self.config.store_router_path,
            port=self.config.store_port,
            store_name=self.config.store_name,
        )
        store.add_documents(
            documents=chunks,
            doc_embeds=embds,
            doc_index=range(len(chunks)),
            doc_name="register-rag",
            store_name="register-rag-eval",
        )

    def load(self) -> Dict[str, List[Dict[str, str | List[str]]]]:
        if self.config.insert:
            self.insert_data()

        qas_path = os.path.join(self.data_path, "qas.json")
        with open(qas_path, "r") as f:
            qas = json.load(f)

        return qas
