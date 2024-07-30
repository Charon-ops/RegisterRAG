from typing import List, Tuple

from .module_factory import (
    EmbeddingFactory,
    StoreFactory,
    ResponseGeneratorFactory,
    PromptGeneratorFactory,
)
from .config import Config
from .documents import Document


class Pipeline:
    """
    A class to create a pipeline for the RAG model. The pipeline consists of the following components:

    1. Embedding: The embedding getter object to get the embeddings of the documents.
    2. Store: The store object to store the documents and their embeddings.
    3. Response Generator: The response generator object to generate the response.
    4. Prompt Generator: The prompt generator object to generate the prompt.
    """

    def __init__(self, config: str | Config) -> None:
        if isinstance(config, str):
            self.config = Config.load(config)
        else:
            self.config = config

        self.embedding = None
        self.store = None
        self.response_generator = None
        self.prompt_generator = None

        self.__init()

    def __init(self) -> None:
        """
        Initialize the pipeline components.
        """
        self.embedding = EmbeddingFactory.create(self.config)
        self.store = StoreFactory.create(self.config)
        self.response_generator = ResponseGeneratorFactory.create(self.config)
        self.prompt_generator = PromptGeneratorFactory.create(self.config)

    async def add_docs(
        self,
        docs: List[Document],
        collection_name: str = "default",
        ids: List[str] = None,
    ) -> None:
        """
        Add the documents to the store

        Args:
            docs (List[Document]): The list of documents to add.
            collection_name (str, optional): The name of the collection to add the documents. Defaults to "default".
            ids (List[str], optional): The list of ids of the documents. If not provided, uuids are generated.
        """
        embeds = await self.embedding.get_embedding(docs)
        await self.store.add_documents(
            documents=docs, embeddings=embeds, collection_name=collection_name, ids=ids
        )

    async def get_response(
        self,
        query: str,
        collection_name: str = "default",
        top_k: int = 5,
        return_related_docs: bool = False,
    ) -> str | Tuple[str, List[Document]]:
        """
        Get the response for the query

        Args:
            query (str): The query to get the response.
            collection_name (str, optional): The name of the collection to search for the query. Defaults to "default".
            top_k (int, optional): The number of related documents to consider. Defaults to 5.
            return_related_docs (bool, optional): Whether to return the related documents or not. Defaults to False.

        Returns:
            str | Tuple[str, List[Document]]: The response for the query. If return_related_docs is True, it
            returns the response and the related documents.
        """
        query_embed = await self.embedding.get_embedding(
            docs=[Document(page_content=query)]
        )
        query_embed = query_embed[0]
        related_docs = await self.store.search_by_embedding(
            embedding=query_embed, collection_name=collection_name, top_k=top_k
        )
        prompt = await self.prompt_generator.get_prompt(query, related_docs)
        response = await self.response_generator.generate(prompt)
        if not return_related_docs:
            return response
        else:
            return response, related_docs

    async def unload(self) -> None:
        """
        Unload the pipeline components. It is not working for now.
        """
        # TODO: 为所有模块添加unload方法
        self.embedding = None
        self.store = None
        self.response_generator = None
        self.prompt_generator = None

        from torch import cuda

        cuda.empty_cache()
