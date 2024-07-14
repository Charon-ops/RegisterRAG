from typing import List
import os
import sqlite3

from langchain_core.documents import Document

from .base import Loader


class SqliteLoader(Loader):
    def __init__(self, db_path: str) -> None:
        if db_path is None or not os.path.isfile(db_path):
            raise ValueError("Invalid weight path provided")
        self.db_path = db_path

    def load_file(self) -> List[Document]:
        conn = sqlite3.connect(self.db_path)
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
                        metadata={"source": f"{self.db_path}::{table_name}"},
                    )
                )

        return documents
