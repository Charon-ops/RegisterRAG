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
