import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from ..config import Config
from ..documents import Document
from ..exceptions.embedding_exceptions import LoadNotInitializedException


class EmbeddingGetter(ABC):
    """
    Base Class: EmbeddingGetter

    Purpose:
        This class serves as a base for deriving specific embedding getter classes used to
        retrieve embeddings for documents.

    Usage:
        - Inherit this class and implement the necessary methods (`embedding` and `load`)
        to adapt to specific models.
        - Call `get_embedding` with a list of `register-rag.entity.Document`
        to retrieve embeddings. This method ensures the model is loaded
        (either waits for `load` to complete or raises an exception if loading fails).

    Methods:
        - `load`: Should be asynchronous (`async`). Initialize the loading task in the
        `__init__` method of the subclass.
        - `embedding`: Define how embeddings are generated from documents.
        - `pre_embedding`: Optional. Define any preprocessing needed before embeddings
        are generated. This method is called before `embedding`.
        - `post_embedding`: Optional. Define any postprocessing after embeddings are
        generated. This method is called immediately after `embedding`,
        but note that `get_embedding` does not wait for its completion
        to allow faster response times.

    Parameters:
        - Pre and post embedding parameters should be passed as dictionaries.
        For example, to pass parameter `a` to the `pre_embedding` method, use:

        ```python
        await embedding_getter.get_embedding(docs, pre_args={"a": 1})
        ```

    Note:
        If using a different embedding strategy, ensure to inherit from this class
        and implement or override the necessary methods as described above.
    """

    def __init__(self, config: Config) -> None:
        """
        Initialize the EmbeddingGetter class.

        Args:
            config (Config): The configuration object for the RAG pipeline.
        """
        super().__init__()
        self.load_task: asyncio.Task = None
        self.after_embedding_task: asyncio.Task = None
        self.config = config

    async def get_embedding(
        self,
        docs: List[Document],
        pre_args: Dict[str, Any] = None,
        after_args: Dict[str, Any] = None,
    ) -> List[List[float]]:
        """A method to get the embeddings of the documents. It should not be overridden if you want to use our default pipeline.

        Args:
            docs (List[Document]): A list of documents to get the embeddings. The content should be stored in the page_content attribute.
            pre_args (Dict[str, Any], optional): Arguments to pass to the pre_embedding method. Defaults to None.
            after_args (Dict[str, Any], optional): Arguments to pass to the after_embedding method. Defaults to None.

        Returns:
            List[List[float]]: A list of embeddings of the documents.

        Raises:
            NotImplementedError: If the method is not implemented in the sub class.
        """
        if self.load_task is None:
            try:
                self.load_task = asyncio.create_task(
                    self.load()
                )  # Create the load task if it is not created.
            except NotImplementedError:
                raise LoadNotInitializedException(
                    self.__class__.__name__
                )  # If the load method is not implemented, raise an exception.

        await self.load_task  # Wait for the load method to be completed.

        if self.after_embedding_task is not None:
            # Wait for the after embedding process to be completed,
            # because it is not awaited in the after_embedding method.
            await self.after_embedding_task

        await self.pre_embedding(
            **(pre_args if pre_args is not None else {})
        )  # Do some pre embedding process if needed.

        # Get the embeddings of the documents.
        res = await self.embedding(docs)

        self.after_embedding_task = asyncio.create_task(
            self.post_embedding(**(after_args if after_args is not None else {}))
        )  # Do some post embedding process if needed, but don't wait for it.

        return res

    async def pre_embedding(self, **kwargs) -> None:
        """A method to do some pre embedding process. It should be implemented in the sub class if needed."""
        return

    @abstractmethod
    async def embedding(self, docs: List[Document]) -> List[List[float]]:
        """The logic to get the embeddings of the documents should be implemented in this method.

        Args:
            docs (List[Document]): A list of documents(`langchain_core.document.Document`) to get the embeddings. The content should be stored in the page_content attribute.

        Raises:
            NotImplementedError: If the method is not implemented in the sub class.

        Returns:
            List[List[float]]: A list of embeddings of the documents.
        """
        raise NotImplementedError(
            "embedding method is not implemented, it should be implemented in the sub class."
        )

    async def post_embedding(self, **kwargs) -> None:
        """A method to do some post embedding process. It should be implemented in the sub class if needed."""
        return

    @abstractmethod
    async def load(self) -> None:
        """Load the embedding model. For remote services, you can test the connection here. The `get_embedding` method will wait for this method to be completed.

        Raises:
            NotImplementedError: If the method is not implemented in the sub class.
        """
        raise NotImplementedError(
            "load method is not implemented, it should be implemented in the sub class."
        )
