from typing import List
from .. import Document
from .splitter import Splitter


class CharacterSplitter(Splitter):
    """
    Split the document into multiple documents by character.
    """

    def __init__(self) -> None:
        super().__init__()

    async def split(
        self, doc: Document, separator: str = "\n", **kargs
    ) -> List[Document]:
        """
        Split a document into multiple documents by character

        Args:
            doc (Document): The document to be split.
            separator (str, optional): The separator to split the document. Defaults to "\\n".

            If some additional metadata is needed, you can pass it in the `metadata` argument.
            For example:

            ```python
            splitter = CharacterSplitter()
            docs = splitter.split(doc, separator="\\n", metadata={"key": "value"})
            ```

        Returns:
            List[Document]: A list of documents after splitting. The metadata of the original
            document will be copied to the new documents.
        """
        metadata = doc.metadata
        if "metadata" in kargs:
            metadata.update(kargs["metadata"])
        chunk_content = doc.page_content.split(separator)
        return [
            Document(page_content=chunk, metadata=metadata) for chunk in chunk_content
        ]
