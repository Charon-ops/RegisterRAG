import os
import chardet

from .loader import Loader
from .. import Document


class TextLoader(Loader):
    def __init__(self, file_path: str) -> None:
        super().__init__(file_path)

    async def load_file(self, file_path: str) -> Document:
        """
        Load a single file from the file path.

        Args:
            file_path (str): The path to the file to load.

        Returns:
            Document: The document loaded from the file. The metadata of the document
            will contain the source file path(`src`) and the encoding(`encoding`) of the file.

            For example:

            ```python
            Document(
                page_content="The content of the file",
                metadata={"src": "path/to/file.txt", "encoding": "utf-8"}
            )
            ```
        """
        assert os.path.isfile(file_path), f"File {file_path} does not exist."
        # if the function is called correctly, the file should exist, and it
        # should be a file

        with open(f"{file_path}", "rb") as f:
            # detect the encoding of the file
            data = f.read(max(512, os.path.getsize(file_path)))
            guess_encoding = chardet.detect(data)["encoding"]

        encoding = guess_encoding if guess_encoding is not None else "utf-8"

        with open(f"{file_path}", "r", encoding=encoding) as f:
            return Document(
                page_content=f.read(), metadata={"src": file_path, "encoding": encoding}
            )
