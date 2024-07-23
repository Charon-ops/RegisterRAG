from typing import List
from register_rag.config.store_config.store_config import StoreConfig
from register_rag.documents.document import Document
from .. import Store


class LocalStore(Store):
    def __init__(self, config: StoreConfig) -> None:
        super().__init__(config)
        if config.store_local_path is None:
            raise ValueError("store_local_path must be set for LocalStore")
