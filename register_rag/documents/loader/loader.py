from abc import ABC, abstractmethod
from typing import List

from .. import Document
from ..splitter import Splitter


class Loader(ABC):
    """
    Base class for all loaders.

    The file path should be passed to the loader.

    If a directory is passed, the loader will load all files in the directory,
    each file will be loaded as a separate document.
    """

    def __init__(self, file_path: str) -> None:
        super().__init__()
        self.file_path = file_path

    @abstractmethod
    async def load(self) -> Document | List[Document]:
        """
        Load the document(s) from the file path.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.

        Returns:
            Document | List[Document]: If a single document is loaded, return a Document object.
            If multiple documents are loaded, return a list of Document objects
        """
        raise NotImplementedError("load method must be implemented in subclass")

    async def load_and_split(self, splitter: Splitter) -> List[Document]:
        """
        Load the document and split it using the splitter.

        Args:
            splitter (Splitter): The splitter to use for splitting the document.

        Returns:
            List[Document]: A list of documents after splitting
        """
        doc = await self.load()
        return await splitter.split(doc)
