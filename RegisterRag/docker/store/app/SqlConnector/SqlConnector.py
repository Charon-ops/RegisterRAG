from abc import ABC, abstractmethod
from typing import List
import threading


class SqlConnector(ABC):
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(SqlConnector, cls).__new__(cls)
                    cls._instance.__initialized = False
        return cls._instance

    def __init__(
        self,
        db_path,
    ):
        if hasattr(self, "__initialized") and self.__initialized:
            return
        self.db_path = db_path
        self.index = None
        self.lock = threading.Lock()
        self.conn = None
        self._setup_database()
        self.__initialized = True

    @abstractmethod
    def _setup_database(self):
        raise NotImplementedError("_setup_database must be implemented in a sub class")

    @abstractmethod
    def add_documents(
        self,
        doc: List[str],
        embedding: List[List[float]],
        in_doc_index: List[int],
        doc_id: int = -1,
    ):
        raise NotImplementedError("add_documents must be implemented in a sub class")
