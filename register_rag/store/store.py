from abc import ABC, abstractmethod
from typing import List

from ..config.store_config import StoreConfig
from ..documents import Document


class Store(ABC):
    """
    Base Class of Store
    """

    def __init__(self, config: StoreConfig) -> None:
        """
        Initialize the store with the given configuration.

        Args:
            config (StoreConfig): The configuration of the store.
        """
        self.config = config

    @abstractmethod
    async def add_document(
        self,
        document: Document,
        embedding: List[float],
        collection_name: str,
        id: str = None,
    ) -> None:
        """Add a document to the store.

        Args:
            document (Document): A document to add.
            embedding (List[float]): An embedding of the document.
            collection_name (str): The name of the collection to add the document.
            id (str, optional): The id of the document. Defaults to None. If it is None,
            it will be generated by using `uuid.uuid4()`.
        """
        raise NotImplementedError("add_document method is not implemented.")

    async def add_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
        collection_name: str,
        ids: List[str] = None,
    ) -> None:
        """
        Add multiple documents to the store.

        Args:
            documents (List[Document]): The documents to add.
            embeddings (List[List[float]]): The embeddings of the documents.
            collection_name (str): The name of the collection to add the documents.
            ids (List[str], optional): The ids of the documents. Defaults to None.
        """
        for doc, embed in zip(documents, embeddings):
            await self.add_document(doc, embed, collection_name, ids)

    @abstractmethod
    async def search_by_embedding(
        self, embedding: List[float], collection_name: str, top_k: int = 5
    ) -> List[Document]:
        """
        Search a document by its embedding.

        Args:
            embedding (List[float]): An embedding of the document.
            collection_name (str): The name of the collection to search the document.
            top_k (int, optional): The number of documents to return. Defaults to 5.
        Raises:
            NotImplementedError: If the method is not implemented in the sub class.

        Returns:
            List[Document]: The documents that have the closest embeddings to the given embedding.
        """
        raise NotImplementedError("search_by_embedding method is not implemented.")

    async def search_by_embeddings(
        self, embeddings: List[List[float]], collection_name: str, top_k: int = 5
    ) -> List[List[Document]]:
        """
        Search documents by their embeddings.

        Args:
            embeddings (List[List[float]]): A list of embeddings of the documents.
            collection_name (str): The name of the collection to search the documents.
            top_k (int, optional): The number of documents to return. Defaults to 5.

        Returns:
            List[List[Document]]: A list of documents that have the closest embeddings to the
            given embeddings.
        """
        return [
            await self.search_by_embedding(embedding, collection_name)
            for embedding in embeddings
        ]

    @abstractmethod
    async def get_id_by_document(self, document: Document, collection_name: str) -> str:
        """
        Get the id of a document.

        Args:
            document (Document): A document to get the id.
            collection_name (str): The name of the collection to get the id.

        Raises:
            NotImplementedError: If the method is not implemented in the sub class.

        Returns:
            str: The id of the document.
        """
        raise NotImplementedError("get_document_by_id method is not implemented.")

    async def get_ids_by_documents(
        self, documents: List[Document], collection_name: str
    ) -> List[str]:
        """
        Get the ids of the documents

        Args:
            documents (List[Document]): A list of documents to get the ids.
            collection_name (str): The name of the collection to get the ids.

        Returns:
            List[str]: A list of ids of the documents.
        """
        return [
            await self.get_id_by_document(doc, collection_name) for doc in documents
        ]

    @abstractmethod
    async def delete_by_id(self, id: str, collection_name: str) -> None:
        """
        Delete a document by its id.

        Args:
            id (str): The id of the document to delete.
            collection_name (str): The name of the collection to delete the document.

        Raises:
            NotImplementedError: If the method is not implemented in the sub class.
        """
        raise NotImplementedError("delete_by_id method is not implemented.")

    async def delete_by_ids(self, ids: List[str], collection_name: str) -> None:
        """
        Delete multiple documents by their ids.

        Args:
            ids (List[str]): The ids of the documents to delete.
            collection_name (str): The name of the collection to delete the documents.
        """
        for id in ids:
            await self.delete_by_id(id, collection_name)