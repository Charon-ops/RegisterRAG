from typing import List
import sqlite3

from ...documents import Document
from .loader import Loader


class SqliteLoader(Loader):
    """
    A loader for SQLite files. The loader uses the `sqlite3` package to load SQLite files.

    The file path should be a string representing the path to the SQLite file. The loader will load
    all tables in the SQLite file as separate documents. One record in the table will be one document.
    """

    def __init__(self, file_path: str) -> None:
        super().__init__(file_path)

    async def load_file(self, file_path: str) -> List[Document]:
        """
        Load a single SQLite file from the file path.

        Args:
            file_path (str): The path to the SQLite file to load.

        Returns:
            List[Document]: The documents loaded from the SQLite file. Each table in the SQLite file
            will be loaded as a separate document. The metadata of the document will contain the source
            file path(`src`) and the table name(`table`). One record in the table will be one document.
        """
        conn = sqlite3.connect(file_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        table_names = [table[0] for table in tables]

        documents = []

        for table_name in table_names:
            # Fetch columns from the table
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            columns = [column[1] for column in columns]

            # Get the contents of the table
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()

            for row in rows:
                formatted_string = ", ".join(
                    f"{col}: {val}" for col, val in zip(columns, row)
                )
                documents.append(
                    Document(
                        page_content=formatted_string,
                        metadata={
                            "src": f"{self.file_path}",
                            "table": f"{table_name}",
                        },
                    )
                )

        return documents
