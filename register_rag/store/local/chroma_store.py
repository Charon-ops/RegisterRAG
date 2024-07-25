from typing import List
import os
import uuid

from ...config import Config
from ...documents import Document
from .local_store import LocalStore


class ChromaStore(LocalStore):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "ChromaStore requires `chromadb`"
                "You can install it with `pip install chromadb`"
            )
        if not os.path.exists(config.store.store_local_path):
            os.makedirs(config.store.store_local_path)
        self.client = chromadb.PersistentClient(config.store.store_local_path)

    async def add_document(
        self,
        document: Document,
        embedding: List[float],
        collection_name: str,
        id: str = None,
    ) -> None:
        collection = self.client.get_or_create_collection(collection_name)
        if not id:
            id = uuid.uuid4()
        collection.add(
            ids=[id],
            embeddings=[embedding],
            documents=[document.page_content],
            metadatas=[document.metadata],
        )

    async def add_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
        collection_name: str,
        ids: List[str] = None,
    ) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings should have the same length.")

        if not ids:
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]

        collection = self.client.get_or_create_collection(collection_name)
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=[doc.page_content for doc in documents],
            metadatas=[doc.metadata for doc in documents],
        )

    async def search_by_embedding(
        self, embedding: List[float], collection_name: str, top_k: int = 5
    ) -> List[Document]:
        collection = self.client.get_or_create_collection(collection_name)
        result = collection.query(query_embeddings=[embedding], n_results=top_k)
        if result["documents"] and result["metadatas"]:
            return [
                Document(page_content=doc, metadata=metadata)
                for doc, metadata in zip(result["documents"][0], result["metadatas"][0])
            ]
        return []

    async def get_id_by_document(self, document: Document, collection_name: str) -> str:
        """
        Get the id of the document.

        Args:
            document (Document): The document to get the id.
            collection_name (str): The name of the collection to search the document.

        Returns:
            str: The id of the document. If not found, return None.
        """
        collection = self.client.get_or_create_collection(collection_name)
        result = collection.get(
            where_document={"$contains": document.page_content},
        )
        if result["ids"]:
            return result["ids"][0]
        return None

    async def delete_by_id(self, id: str, collection_name: str) -> None:
        collection = self.client.get_or_create_collection(collection_name)
        collection.delete(ids=[id])

    async def delete_by_ids(self, ids: List[str], collection_name: str) -> None:
        collection = self.client.get_or_create_collection(collection_name)
        collection.delete(ids=ids)
