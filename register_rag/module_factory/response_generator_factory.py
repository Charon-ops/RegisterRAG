from ..config import Config
from ..response_generators import Generator
from ..response_generators.local import TransformersGenerator
from ..response_generators.remote import OllamaGenerator, XinferenceGenerator


class ResponseGeneratorFactory:
    """
    A factory class to create the response generator object based on the configuration.
    """

    __local_generator_map = {
        "transformers": TransformersGenerator,
    }

    __remote_generator_map = {
        "ollama": OllamaGenerator,
        "xinference": XinferenceGenerator,
    }

    @classmethod
    def create(cls, config: Config) -> Generator:
        """
        Create the response generator object.

        Args:
            config (Config): The configuration object for the RAG pipeline.

        Raises:
            ValueError: If the generator type is not 'local' or 'remote'.

        Returns:
            Generator: The response generator object.
        """
        generator_config = config.generation
        if generator_config.generation_type not in ["local", "remote"]:
            raise ValueError(
                "Invalid generator type. Please choose 'local' or 'remote'."
            )
        if generator_config.generation_type == "local":
            return cls.__create_local_generator(config)
        else:
            return cls.__create_remote_generator(config)

    @classmethod
    def get_class_name_from_config(cls, config: Config) -> str:
        """
        Get the class name of the response generator from the configuration

        Args:
            config (Config): The configuration object for the RAG pipeline.

        Raises:
            ValueError: If the generator type is not 'local' or 'remote'.

        Returns:
            str: The class name of the response generator.
        """
        if config.generation.generation_type not in ["local", "remote"]:
            raise ValueError(
                "Invalid generator type. Please choose 'local' or 'remote'."
            )
        if config.generation.generation_type == "local":
            generator_name = config.generation.generation_model_name_or_path.split("/")[
                0
            ]
            return cls.__local_generator_map[generator_name].__name__
        else:
            generator_name = config.generation.generation_model_name_or_path.split("/")[
                0
            ]
            return cls.__remote_generator_map[generator_name].__name__

    @classmethod
    def get_config_name_from_class_name(cls, class_name: str) -> str:
        """
        Get the configuration name of the response generator from the class name.

        Args:
            class_name (str): The class name of the response generator.

        Raises:
            ValueError: If no class with the given name is found.

        Returns:
            str: The configuration name of the response generator.
        """
        for k in cls.__local_embedding_map:
            if cls.__local_embedding_map[k].__name__ == class_name:
                return k
        for k in cls.__remote_embedding_map:
            if cls.__remote_embedding_map[k].__name__ == class_name:
                return k
        raise ValueError(f"No class with name {class_name} found.")

    @classmethod
    def __create_local_generator(cls, config: Config) -> Generator:
        """
        Create a local generator based on the configuration.

        Args:
            config (Config): The configuration object for the RAG pipeline.

        Raises:
            ValueError: If the generator name is not found in the local generator map.

        Returns:
            Generator: The local generator object.
        """
        generator_name = config.generation.generation_model_name_or_path.split("/")[0]
        if generator_name not in cls.__local_generator_map:
            raise ValueError(
                f"No local generator available for model: {generator_name}"
            )
        return cls.__local_generator_map[generator_name](config)

    @classmethod
    def __create_remote_generator(cls, config: Config) -> Generator:
        """
        Create a remote generator based on the configuration.

        Args:
            config (Config): The configuration object for the RAG pipeline.

        Raises:
            ValueError: If the generator name is not found in the remote generator map.

        Returns:
            Generator: The remote generator object.
        """
        generator_name = config.generation.generation_model_name_or_path.split("/")[0]
        if generator_name not in cls.__remote_generator_map:
            raise ValueError(
                f"No remote generator available for model: {generator_name}"
            )
        return cls.__remote_generator_map[generator_name](config)
