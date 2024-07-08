import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from ..entity import Document
from ..exceptions.embedding_exceptions import LoadNotInitializedException


class EmbeddingGetter(ABC):
    """Base class for embedding getter.

    This class should be inherited by the class that will be used to get the embeddings.

    You can just call the get_embedding method to get the embeddings of the documents.
    The documents should be a `list` of `register-rag.entity.Document`.
    It will check if the model is finished correctly, and if not,
    it will wait the load method to load the model or raise an exception.

    If you want to use a different embedding getter,
    you can just inherit this class and implement the embedding and load method.
    The load method shuold be `async` and the task
    should be created in the `__init__` method.

    If some pre embedding process is needed, you can implement the pre_embedding method.
    It will be called before the embedding method.

    If some after embedding process is needed, you can implement the after_embedding method.
    It will be called after the embedding method. But it should be remembered that
    the get_embedding method will not wait for this method to be completed.

    The parameters of the pre_embedding and after_embedding should be passed as a dictionary.
    For example, if you want to pass a parameter `a` to the pre_embedding method,
    you should call the get_embedding method like this:
    ```
    await embedding_getter.get_embedding(docs, pre_args={"a": 1})
    ```
    """

    def __init__(self) -> None:
        self.load_task: asyncio.Task = None
        self.after_embedding_task: asyncio.Task = None

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
        assert (
            self.load_task is not None,
            LoadNotInitializedException(self.__class__.__name__),
        )  # Make sure the load method is implemented correctly.

        await self.load_task  # Wait for the load method to be completed.

        if self.after_embedding_task is not None:
            await self.after_embedding_task  # Wait for the after embedding process to be completed.

        await self.pre_embedding(
            **(pre_args if pre_args is not None else {})
        )  # Do some pre embedding process if needed.

        res = await self.embedding(docs)  # Get the embeddings of the documents.

        self.after_embedding_task = asyncio.create_task(
            self.after_embedding(**(after_args if after_args is not None else {}))
        )  # Do some post embedding process if needed, but don't wait for it.

        return res

    @abstractmethod
    async def pre_embedding(self, **kwargs) -> None:
        """A method to do some pre embedding process. It should be implemented in the sub class if needed."""
        return

    @abstractmethod
    async def embeddng(self, docs: List[Document]) -> List[List[float]]:
        """The logic to get the embeddings of the documents should be implemented in this method.

        Args:
            docs (List[Document]): A list of documents(`langchain_core.document.Document`) to get the embeddings. The content should be stored in the page_content attribute.

        Raises:
            NotImplementedError: If the method is not implemented in the sub class.

        Returns:
            List[List[float]]: A list of embeddings of the documents.
        """
        raise NotImplementedError(
            "_get_embedding method is not implemented, it should be implemented in the sub class."
        )

    @abstractmethod
    async def after_embedding(self, **kwargs) -> None:
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
