from ..config import Config
from ..embeddings import EmbeddingGetter
from ..embeddings.local import BertEmbeddingGetter, SentenceTransformerEmbeddingGetter
from ..embeddings.remote import XinferenceEmbeddingGetter


class EmbeddingFactory:
    """
    A factory class to create the embedding getter object.

    You can use the `create` method to create the embedding getter object.
    """

    __local_embedding_map = {
        "bert": BertEmbeddingGetter,
        "sentence_transformer": SentenceTransformerEmbeddingGetter,
    }

    __remote_embedding_map = {
        "xinference": XinferenceEmbeddingGetter,
    }

    @classmethod
    def create(cls, config: Config) -> EmbeddingGetter:
        """
        Create the embedding getter object.

        Args:
            config (Config): The configuration object for the RAG pipeline.

        Raises:
            ValueError: If the embedding type is not 'local' or 'remote'.

        Returns:
            EmbeddingGetter: The embedding getter object.
        """
        embedding_config = config.embedding
        if embedding_config.embedding_type not in ["local", "remote"]:
            raise ValueError(
                "Invalid embedding type. Please choose 'local' or 'remote'."
            )

        if embedding_config.embedding_type == "local":
            return cls.__create_local_embedding(config)
        else:
            return cls.__create_remote_embedding(config)

    @classmethod
    def get_class_name_from_config(cls, config: Config) -> str:
        """
        Get the class name of the embedding getter from the configuration

        Args:
            config (Config): The configuration object for the RAG pipeline.

        Raises:
            ValueError: If the embedding type is not 'local' or 'remote'.

        Returns:
            str: The class name of the embedding getter.
        """
        if config.embedding.embedding_type not in ["local", "remote"]:
            raise ValueError(
                "Invalid embedding type. Please choose 'local' or 'remote'."
            )
        if config.embedding.embedding_type == "local":
            return (
                cls.__local_embedding_map[
                    config.embedding.embedding_model_name_or_path.split("/")[0]
                ]
            ).__name__
        else:
            return (
                cls.__remote_embedding_map[
                    config.embedding.embedding_model_name_or_path.split("/")[0]
                ]
            ).__name__

    @classmethod
    def get_config_name_from_class_name(cls, class_name: str) -> str:
        """
        Get the configuration name of the embedding getter from the class name

        Args:
            class_name (str): The class name of the embedding getter.

        Raises:
            ValueError: If the class name is not found in the local or remote embedding map.

        Returns:
            str: The configuration name of the embedding getter.
        """
        for k in cls.__local_embedding_map:
            if cls.__local_embedding_map[k].__name__ == class_name:
                return k
        for k in cls.__remote_embedding_map:
            if cls.__remote_embedding_map[k].__name__ == class_name:
                return k
        raise ValueError(f"No class with name {class_name} found.")

    @classmethod
    def __create_local_embedding(cls, config: Config) -> EmbeddingGetter:
        """
        Create a local embedding getter object.

        Args:
            config (Config): The configuration object for the RAG pipeline.

        Raises:
            ValueError: If the model name is not found in the local embedding map.

        Returns:
            EmbeddingGetter: The local embedding getter object.
        """
        model_name = config.embedding.embedding_model_name_or_path.split("/")[0]
        if model_name not in cls.__local_embedding_map:
            raise ValueError(
                f"No local embedding getter available for model: {model_name}"
            )
        return cls.__local_embedding_map[model_name](config)

    @classmethod
    def __create_remote_embedding(cls, config: Config) -> EmbeddingGetter:
        """
        Create a remote embedding getter object.

        Args:
            config (Config): The configuration object for the RAG pipeline.

        Raises:
            ValueError: If the model name is not found in the remote embedding map.

        Returns:
            EmbeddingGetter: The remote embedding getter object.
        """
        model_name = config.embedding.embedding_model_name_or_path.split("/")[0]
        if model_name not in cls.__remote_embedding_map:
            raise ValueError(
                f"No remote embedding getter available for model: {model_name}"
            )
        return cls.__remote_embedding_map[model_name](config)
