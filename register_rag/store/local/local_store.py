from typing import List
from ...config import Config
from .. import Store


class LocalStore(Store):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if self.config.store.store_name is None:
            raise ValueError("store_name must be set for LocalStore")
        if self.config.store.store_local_path is None:
            raise ValueError("store_local_path must be set for LocalStore")
