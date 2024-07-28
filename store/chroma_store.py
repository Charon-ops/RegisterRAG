from typing import List
import uuid

import chromadb
from chromadb import QueryResult
from langchain_core.documents import Document
from xinference_client import RESTfulClient

from .base import Store


class ChromaStoreEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, url: str) -> None:
        super().__init__()
        self.url = url

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        client = RESTfulClient(self.url)
        model = client.get_model("bge-m3")
        return [item["embedding"] for item in model.create_embedding(input)["data"]]


class ChromaStore(Store):
    def __init__(self, db_path: str):
        self.client = chromadb.PersistentClient(db_path)

    def add_documents(
        self,
        documents: List[Document],
        embedding_remote_url: str,
        collection_name: str = "default",
        from_sql: bool = False,
    ) -> None:
        if len(documents) == 0 or documents is None:
            return
        try:
            collection = self.client.get_collection(
                collection_name,
                embedding_function=ChromaStoreEmbeddingFunction(embedding_remote_url),
            )
        except ValueError:
            collection = self.client.create_collection(
                name=collection_name,
                embedding_function=ChromaStoreEmbeddingFunction(embedding_remote_url),
            )
        if not from_sql:
            collection.add(
                documents=[d.page_content for d in documents],
                metadatas=[d.metadata for d in documents],
                ids=[str(uuid.uuid4()) for i in range(len(documents))],
            )
        else:
            embedding_funtion = ChromaStoreEmbeddingFunction(embedding_remote_url)
            embeddings = embedding_funtion([d.page_content for d in documents])
            embedding = [sum(x) / len(x) for x in zip(*embeddings)]
            content = ""
            for doc in documents:
                content += doc.page_content + "\n"
            collection.add(
                documents=[content],
                embeddings=[embedding],
                metadatas=[documents[0].metadata],
                ids=[str(uuid.uuid4())],
            )

    def search_by_embed(
        self,
        query_embed: List[List[float]],
        collction_name: str = "default",
        results: int = 10,
    ) -> QueryResult:
        try:
            collection = self.client.get_collection(name=collction_name)
        except ValueError:
            return []
        return collection.query(query_embeddings=query_embed, n_results=results)

    def get_id_by_docs(
        self, docs: List[Document], collection_name: str = "default"
    ) -> List[str]:
        try:
            collection = self.client.get_collection(name=collection_name)
        except ValueError:
            return []
        ids = []
        for doc in docs:
            ids.append(
                collection.query(where_document={"$contains": doc.page_content})["ids"][
                    0
                ]
            )
        return ids

    def delete_documents_by_ids(self, doc_ids: List[int], collection_name: str) -> bool:
        try:
            collection = self.client.get_collection(name=collection_name)
        except ValueError:
            return
        collection.delete(ids=doc_ids)