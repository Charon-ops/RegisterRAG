from ..config import Config
from ..store import Store
from ..store.local import ChromaStore


class StoreFactory:
    """
    A factory class to create the store object based on the configuration.
    """

    __local_store_map = {
        "chroma": ChromaStore,
    }

    __remote_store_map = {}

    @classmethod
    def create(cls, config: Config) -> Store:
        """
        Create the store object.

        Args:
            config (Config): The configuration object for the RAG pipeline.

        Raises:
            ValueError: If the store type is not 'local' or 'remote'.

        Returns:
            Store: The store object.
        """
        store_config = config.store
        if store_config.store_type not in ["local", "remote"]:
            raise ValueError("Invalid store type. Please choose 'local' or 'remote'.")
        if store_config.store_type == "local":
            return cls.__create_local_store(config)
        else:
            return cls.__create_remote_store(config)

    @classmethod
    def __create_local_store(cls, config: Config) -> Store:
        """
        Create the local store object

        Args:
            config (Config): The configuration object for the RAG pipeline.

        Raises:
            ValueError: If the store type is not 'local' or 'remote'.

        Returns:
            Store: The store object.
        """
        store_name = config.store.store_name.split("/")[0]
        if store_name not in cls.__local_store_map:
            raise ValueError(f"No local store available for model: {store_name}")
        return cls.__local_store_map[store_name](config)

    @classmethod
    def get_class_name_from_config(cls, config: Config) -> str:
        """
        Get the class name of the store object based on the configuration.

        Args:
            config (Config): The configuration object for the RAG pipeline.

        Raises:
            ValueError: If the store type is not 'local' or 'remote'.

        Returns:
            str: The class name of the store object.
        """
        if config.store.store_type not in ["local", "remote"]:
            raise ValueError("Invalid store type. Please choose 'local' or 'remote'.")
        if config.store.store_type == "local":
            return cls.__local_store_map[config.store.store_name.split("/")[0]].__name__
        else:
            return cls.__remote_store_map[
                config.store.store_name.split("/")[0]
            ].__name__

    @classmethod
    def get_config_name_from_class_name(cls, class_name: str) -> str:
        """
        Get the configuration name of the store object from the class name.

        Args:
            class_name (str): The class name of the store object.

        Raises:
            ValueError: If no class with the given name is found.

        Returns:
            str: The configuration name of the store object.
        """
        for k in cls.__local_embedding_map:
            if cls.__local_embedding_map[k].__name__ == class_name:
                return k
        for k in cls.__remote_embedding_map:
            if cls.__remote_embedding_map[k].__name__ == class_name:
                return k
        raise ValueError(f"No class with name {class_name} found.")

    @classmethod
    def __create_remote_store(cls, config: Config) -> Store:
        """
        Create the remote store object

        Args:
            config (Config): The configuration object for the RAG pipeline.

        Raises:
            ValueError: If the store type is not 'local' or 'remote'.

        Returns:
            Store: The store object.
        """
        store_name = config.store.store_name.split("/")[0]
        if store_name not in cls.__remote_store_map:
            raise ValueError(f"No remote store available for model: {store_name}")
        return cls.__remote_store_map[store_name](config)
