from abc import ABC, abstractmethod
from typing import List

from .. import Document


class Splitter(ABC):
    """
    Base class for all splitters.

    If you want to split a document into multiple documents,
    you should inherit from this class, and implement the split method.

    When you implement the split method, remember to copy the metadata
    from the original document to the new documents if metadata is needer.
    """

    def __init__(self) -> None:
        """
        Initialize the Splitter object.
        """
        super().__init__()

    @abstractmethod
    async def split(self, doc: Document, **kargs) -> List[Document]:
        """
        Split a document into multiple documents.

        Args:
            doc (Document): The document to be split.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.

        Returns:
            List[Document]: A list of documents after splitting.
        """
        raise NotImplementedError("split method must be implemented in subclass")
