from typing import List
import os
import numpy as np
from annoy import AnnoyIndex
from scipy.spatial.distance import cosine

from .Store import Store
from SqlConnector import SqliteConnector


class AnnoyStore(Store):

    def __init__(
        self,
        index_path: str = None,
        db_path: str = None,
        dis_type: str = "angular",
        emb_len: int = 1024,
    ):
        super().__init__(index_path)
        self.index_path = (
            index_path
            if index_path
            else os.path.join(os.path.dirname(__file__), "..", "data", "store.ann")
        )
        self.db_path = (
            db_path
            if db_path
            else os.path.join(os.path.dirname(__file__), "..", "data", "store.db")
        )
        self.dis_type = dis_type
        self.emb_len = emb_len

    def _load(self) -> None:
        with self.lock:
            self.index = AnnoyIndex(self.emb_len, self.dis_type)
            if os.path.exists(self.index_path):
                self.index.load(self.index_path)
            else:
                self.index = None

    def _merge_chunk(self, chunk_id: int, query_embd: List[float]) -> str:
        connector = SqliteConnector(self.db_path)
        doc_id = connector.get_doc_id_by_chunk_id(chunk_id)
        tree = connector.read_tree(doc_id)

        current_emb = connector.get_embedding_by_chunk_id(chunk_id)
        current_dis = cosine(query_embd, current_emb)

        res = connector.get_content_by_chunk_id(chunk_id)
        can_merge = True
        min_index = connector.get_index_by_chunk_id(chunk_id)
        max_index = min_index

        while can_merge:
            min_node = tree.find_by_index(min_index)
            max_node = tree.find_by_index(max_index)
            predecessor_node = (
                tree.get_predecessor(min_node) if min_node is not None else None
            )
            successor_node = (
                tree.get_successor(max_node) if max_node is not None else None
            )

            predecessor_emb = (
                connector.get_embedding_by_chunk_id(predecessor_node.id)
                if predecessor_node is not None
                else None
            )
            successor_emb = (
                connector.get_embedding_by_chunk_id(successor_node.id)
                if successor_node is not None
                else None
            )

            with_predecessor_dis = (
                cosine(
                    query_embd,
                    (
                        (np.array(predecessor_emb) + np.array(current_emb)) / 2.0
                    ).tolist(),
                )
                if predecessor_emb is not None
                else 2.0
            )
            with_successor_dis = (
                cosine(
                    query_embd,
                    ((np.array(successor_emb) + np.array(current_emb)) / 2.0).tolist(),
                )
                if successor_emb is not None
                else 2.0
            )
            with_pre_and_successor_dis = (
                cosine(
                    query_embd,
                    (
                        (
                            np.array(predecessor_emb)
                            + np.array(current_emb)
                            + np.array(successor_emb)
                        )
                        / 3.0
                    ).tolist(),
                )
                if predecessor_emb is not None and successor_emb is not None
                else 2.0
            )

            min_dis = min(
                current_dis,
                with_predecessor_dis,
                with_successor_dis,
                with_pre_and_successor_dis,
            )

            current_dis = min_dis

            if min_dis == with_pre_and_successor_dis:
                res = (
                    connector.get_content_by_chunk_id(predecessor_node.id)
                    + "\n"
                    + res
                    + "\n"
                    + connector.get_content_by_chunk_id(successor_node.id)
                )
                min_index = predecessor_node.index
                max_index = successor_node.index
            elif min_dis == with_predecessor_dis:
                res = (
                    connector.get_content_by_chunk_id(predecessor_node.id) + "\n" + res
                )
                min_index = predecessor_node.index
            elif min_dis == with_successor_dis:
                res = res + "\n" + connector.get_content_by_chunk_id(successor_node.id)
                max_index = successor_node.index
            else:
                can_merge = False
        return res

    def add_documents(
        self,
        doc_list: List[str],
        doc_emb_list: List[List[float]],
        doc_index: List[int],
        doc_name: str = None,
        doc_id: int = None,
    ) -> None:
        assert (
            doc_name is not None or doc_id is not None
        ), "doc_name or doc_id must not be None when add documents"

        connector = SqliteConnector(self.db_path)
        with self.lock:
            connector.add_documents(
                doc=doc_list,
                embedding=doc_emb_list,
                in_doc_index=doc_index,
                doc_id=doc_id,
                doc_name=doc_name,
            )
        self.index = AnnoyIndex(self.emb_len, self.dis_type)
        emb_info = connector.get_emb_info()
        with self.lock:
            for info in emb_info:
                self.index.add_item(info["id"], info["embedding"])
            self.index.build(10)
            self.index.save(self.index_path)

    def search_by_embedding(self, query_emb: List[float], nums: int = 50) -> List[str]:
        self._load()

        with self.lock:
            if self.index is None:
                return []

        search_res = self.index.get_nns_by_vector(
            query_emb, nums, include_distances=True
        )
        ids = search_res[0]
        res = []
        for id in ids:
            res.append(self._merge_chunk(id, query_emb))
        return res

    def delete_by_id(self, chunk_id: int) -> None:
        connector = SqliteConnector(self.db_path)
        if not connector.delete_by_id(chunk_id):
            return
        self.index = AnnoyIndex(self.emb_len, self.dis_type)
        emb_info = connector.get_emb_info()
        with self.lock:
            for info in emb_info:
                self.index.add_item(info["id"], info["embedding"])
            self.index.build(10)
            self.index.save(self.index_path)

    def get_id_by_doc(self, doc: str) -> int:
        connector = SqliteConnector(self.db_path)
        return connector.get_id_by_doc(doc)
