from ..config import Config
from ..response_generators import Generator
from ..response_generators.local import TransformersGenerator
from ..response_generators.remote import OllamaGenerator, XinferenceGenerator


class ResponseGeneratorFactory:
    __local_generator_map = {
        "transformers": TransformersGenerator,
    }

    __remote_generator_map = {
        "ollama": OllamaGenerator,
        "xinference": XinferenceGenerator,
    }

    @classmethod
    def create(cls, config: Config) -> Generator:
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
    def __create_local_generator(cls, config: Config) -> Generator:
        generator_name = config.generation.generation_model_name_or_path.split("/")[0]
        if generator_name not in cls.__local_generator_map:
            raise ValueError(
                f"No local generator available for model: {generator_name}"
            )
        return cls.__local_generator_map[generator_name](config)

    @classmethod
    def __create_remote_generator(cls, config: Config) -> Generator:
        generator_name = config.generation.generation_model_name_or_path.split("/")[0]
        if generator_name not in cls.__remote_generator_map:
            raise ValueError(
                f"No remote generator available for model: {generator_name}"
            )
        return cls.__remote_generator_map[generator_name](config)
