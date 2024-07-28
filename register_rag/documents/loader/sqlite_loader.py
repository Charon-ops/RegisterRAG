import sqlite3

from ...documents import Document
from .loader import Loader


class SqliteLoader(Loader):
    def __init__(self, file_path: str) -> None:
        super().__init__(file_path)

    async def load_file(self, file_path: str) -> Document:
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
