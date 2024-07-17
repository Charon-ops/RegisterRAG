from abc import ABC, abstractmethod
from typing import List, Dict, Any
import asyncio
import os

from .. import Document
from ..splitter import Splitter


class Loader(ABC):
    """
    Base class for all loaders.

    The file path should be passed to the loader. After create an instance of the loader,
    you can call the `load` method to load the document(s) from the file path.

    If a directory is passed, the loader will load all files in the directory,
    each file will be loaded as a separate document. But the loader will not load files
    in subdirectories.

    To support different file types, you can implement the `load_file` method.
    If some pre or post process is needed, you can override the `pre_load` and `post_load`
    methods.

    The `load` method will never wait for the `post_load` method to be completed.
    If you want to wait for the `post_load` method to be completed, you should override
    the `load` method.
    """

    def __init__(self, file_path: str) -> None:
        super().__init__()
        self.file_path = file_path
        self.post_load_task: asyncio.Task = None

    async def load(
        self,
        pre_load_args: Dict[str, Any] = None,
        post_load_args: Dict[str, Any] = None,
    ) -> List[Document]:
        """
        Load the document(s) from the file path.

        Returns:
            List[Document]: A list of documents loaded from the file path. The length
            of the list will equal to the number of files loaded.
        """
        await self.pre_load(**(pre_load_args if pre_load_args is not None else {}))

        if self.post_load_task is not None:
            await self.post_load_task

        res = []

        if os.path.isdir(self.file_path):
            for file in os.listdir(self.file_path):
                if os.path.isfile(os.path.join(self.file_path, file)):
                    res.append(await self.load_file(os.path.join(self.file_path, file)))
        else:
            res.append(await self.load_file(self.file_path))

        self.post_load_task = asyncio.create_task(
            self.post_load(**(post_load_args if post_load_args is not None else {}))
        )

        return res

    @abstractmethod
    async def load_file(self, file_path: str) -> Document:
        """
        Load a single file from the file path.

        Args:
            file_path (str): The path to the file to load.

        Returns:
            Document: The document loaded from the file.
        """
        raise NotImplementedError("load_file method must be implemented in subclass")

    async def pre_load(self, **kargs) -> None:
        """
        Preload the document(s) from the file path.
        """
        pass

    async def post_load(self, **kargs) -> None:
        """
        Postload the document(s) from the file path.
        """
        pass

    async def load_and_split(
        self,
        splitter: Splitter,
        pre_load_args: Dict[str, Any] = None,
        post_load_args: Dict[str, Any] = None,
    ) -> List[Document]:
        """
        Load the document and split it using the splitter.

        Args:
            splitter (Splitter): The splitter to use for splitting the document.

        Returns:
            List[Document]: A list of documents after splitting
        """
        docs = await self.load(pre_load_args, post_load_args)
        res = []
        for doc in docs:
            res.extend(await splitter.split(doc))
