from abc import ABC, abstractmethod
from typing import List
import threading


class Store(ABC):
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(Store, cls).__new__(cls)
                    cls._instance.__initialized = False
        return cls._instance

    def __init__(
        self,
        index_path: str = None,
    ):
        if hasattr(self, "__initialized") and self.__initialized:
            return
        self.index_path = index_path
        self.lock = threading.Lock()
        self.__initialized = True

    @abstractmethod
    def add_documents(
        self, doc_list: List[str], doc_emb_list: List[List[float]]
    ) -> None:
        raise NotImplementedError("add_documents must be implemented in a sub class")

    @abstractmethod
    def search_by_embedding(self, query_emb: List[float], nums: int = 50) -> List[str]:
        raise NotImplementedError(
            "search_by_embedding must be implemented in a sub class"
        )

    def search_by_embeddings(
        self, query_embs: List[List[float]], nums: int = 50
    ) -> List[List[str]]:
        return [self.search_by_embedding(qry_emb, nums) for qry_emb in query_embs]

    @abstractmethod
    def get_id_by_doc(self, doc: str) -> int:
        raise NotImplementedError("get_id_by_doc must be implemented in a sub class")

    @abstractmethod
    def delete_by_id(self, doc_id: int) -> None:
        raise NotImplementedError("delete_by_id must be implemented in a sub class")
