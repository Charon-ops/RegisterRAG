from typing import List, Dict
import threading
import sqlite3
import pickle

from DocTree import DocTree, DocNode
from .SqlConnector import SqlConnector


class SqliteConnector(SqlConnector):
    # TODO: 多线程适配
    def _setup_database(self) -> None:
        with self.lock:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT,
                    in_doc_index INTEGER,
                    embedding BLOB,
                    parent_id INTEGER,
                    left_child_id INTEGER,
                    right_child_id INTEGER,
                    FOREIGN KEY (left_child_id) REFERENCES chunks(id),
                    FOREIGN KEY (right_child_id) REFERENCES chunks(id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id INTEGER,
                    name TEXT,
                    FOREIGN KEY (chunk_id) REFERENCES chunk(id)
                )
                """
            )
            self.conn.commit()

    def _read_tree(self, chunk_id: int) -> DocNode:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
        chunk = cursor.fetchone()
        node = DocNode(chunk_id, chunk[2])
        node.left = self._read_tree(chunk[5]) if chunk[5] else None
        node.right = self._read_tree(chunk[6]) if chunk[6] else None
        if node.left:
            node.left.parent = node
        if node.right:
            node.right.paremt = node
        return node

    def read_tree(self, doc_id: int) -> DocTree:
        cursor = self.conn.cursor()
        chunk_id_search = cursor.execute(
            "SELECT chunk_id FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        chunk_id = chunk_id_search[0] if chunk_id_search else None
        if not chunk_id:
            return DocTree(doc_id)
        root = self._read_tree(chunk_id)
        tree = DocTree(doc_id)
        tree.root = root
        return tree

    def _save_tree(self, node: DocNode, parent_id: int) -> None:
        cursor = self.conn.cursor()
        with self.lock:
            cursor.execute(
                "UPDATE chunks SET parent_id = ?, left_child_id = ?, right_child_id = ? WHERE id = ?",
                (
                    parent_id if parent_id else None,
                    node.left.id if node.left else None,
                    node.right.id if node.right else None,
                    node.id,
                ),
            )
            self.conn.commit()
        if node.left:
            self._save_tree(node.left, node.id)
        if node.right:
            self._save_tree(node.right, node.id)

    def save_tree(self, tree: DocTree) -> None:
        if tree.root.id is None:
            return
        cursor = self.conn.cursor()
        cursor.execute("SELECT chunk_id FROM documents WHERE id = ?", (tree.doc_id,))
        self._save_tree(tree.root, None)
        self.conn.commit()

    def add_documents(
        self,
        doc: List[str],
        embedding: List[List[float]],
        in_doc_index: List[int],
        doc_id: int = None,
        doc_name: str = None,
    ) -> int:
        assert len(doc) == len(
            embedding
        ), "doc and embedding should have the same length"

        cursor = self.conn.cursor()

        if doc_id:
            doc_exists = cursor.execute(
                "SELECT id FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
            if not doc_exists:
                raise ValueError(f"No document found with id {doc_id}")
            avl_tree = self.read_tree(doc_id)
        elif doc_name:
            doc_exists = cursor.execute(
                "SELECT id FROM documents WHERE name = ?", (doc_name,)
            ).fetchone()
            if doc_exists:
                doc_id = doc_exists[0]
                avl_tree = self.read_tree(doc_id)
            else:
                cursor.execute("INSERT INTO documents (name) VALUES (?)", (doc_name,))
                doc_id = cursor.lastrowid
                avl_tree = DocTree(doc_id)
        else:
            raise ValueError("doc_id or doc_name should be provided")

        for i, (text, emb, in_doc_idx) in enumerate(zip(doc, embedding, in_doc_index)):
            chunk_exist = cursor.execute(
                "SELECT id FROM chunks WHERE in_doc_index = ?", (in_doc_idx,)
            ).fetchone()
            if chunk_exist:
                continue
            cursor.execute(
                "INSERT INTO chunks (content, in_doc_index, embedding) VALUES (?, ?, ?)",
                (text, in_doc_idx, sqlite3.Binary(pickle.dumps(emb))),
            )
            chunk_id = cursor.lastrowid
            avl_tree.insert(chunk_id, in_doc_idx)

        cursor.execute(
            "UPDATE documents SET chunk_id = ? WHERE id = ?", (avl_tree.root.id, doc_id)
        )
        self.conn.commit()

        self.save_tree(avl_tree)

        return doc_id

    def get_doc_id_by_chunk_id(self, chunk_id: int) -> int | None:
        cursor = self.conn.cursor()
        root_chunk_id = chunk_id
        continue_search = True
        while continue_search:
            parent_search = cursor.execute(
                "SELECT parent_id FROM chunks WHERE id = ?", (root_chunk_id,)
            ).fetchone()
            if not parent_search or not parent_search[0]:
                continue_search = False
            else:
                root_chunk_id = parent_search[0]
        doc_search = cursor.execute(
            "SELECT id FROM documents WHERE chunk_id = ?", (root_chunk_id,)
        ).fetchone()
        return doc_search[0] if doc_search else None

    def get_content_by_chunk_id(self, chunk_id: int) -> str | None:
        cursor = self.conn.cursor()
        doc_search = cursor.execute(
            "SELECT content FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        return doc_search[0] if doc_search else None

    def get_index_by_chunk_id(self, chunk_id: int) -> int | None:
        cursor = self.conn.cursor()
        doc_search = cursor.execute(
            "SELECT in_doc_index FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        return doc_search[0] if doc_search else None

    def get_embedding_by_chunk_id(self, chunk_id: int) -> List[float] | None:
        cursor = self.conn.cursor()
        embedding_search = cursor.execute(
            "SELECT embedding FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        return pickle.loads(embedding_search[0]) if embedding_search else None

    def get_emb_info(self) -> List[Dict]:
        cursor = self.conn.cursor()
        emb_info = cursor.execute("SELECT id, embedding FROM chunks").fetchall()
        if not emb_info:
            return []
        return [
            {"id": info[0], "embedding": pickle.loads(info[1])} for info in emb_info
        ]

    def get_id_by_doc(self, doc: str) -> int:
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM chunks WHERE content = ?", (doc,))
        exist = cursor.fetchone()
        if exist is None:
            return -1
        return exist[0]

    def delete_by_id(self, chunk_id: int) -> bool:
        cursor = self.conn.cursor()
        with self.lock:
            doc_id = self.get_doc_id_by_chunk_id(chunk_id)
            if not doc_id:
                return False
            tree = self.read_tree(doc_id)
            tree.delete(self.get_index_by_chunk_id(chunk_id))

        self._save_tree(tree.root, None)

        with self.lock:
            cursor.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
            cursor.execute(
                "UPDATE documents SET chunk_id = ? WHERE id = ?", (tree.root.id, doc_id)
            )
            self.conn.commit()
        return True
