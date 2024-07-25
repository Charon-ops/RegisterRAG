from ..config import Config
from ..store import Store
from ..store.local import ChromaStore


class StoreFactory:
    __local_store_map = {
        "chroma": ChromaStore,
    }

    __remote_store_map = {}

    @classmethod
    def create(cls, config: Config) -> Store:
        store_config = config.store
        if store_config.store_type not in ["local", "remote"]:
            raise ValueError("Invalid store type. Please choose 'local' or 'remote'.")
        if store_config.store_type == "local":
            return cls.__create_local_store(config)
        else:
            return cls.__create_remote_store(config)

    @classmethod
    def __create_local_store(cls, config: Config) -> Store:
        store_name = config.store.store_name.split("/")[0]
        if store_name not in cls.__local_store_map:
            raise ValueError(f"No local store available for model: {store_name}")
        return cls.__local_store_map[store_name](config)

    @classmethod
    def __create_remote_store(cls, config: Config) -> Store:
        store_name = config.store.store_name.split("/")[0]
        if store_name not in cls.__remote_store_map:
            raise ValueError(f"No remote store available for model: {store_name}")
        return cls.__remote_store_map[store_name](config)
