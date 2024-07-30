from ..config import Config
from ..prompt_generators import PromptGenerator, SplicePromptGenerator


class PromptGeneratorFactory:
    """
    A factory class to create the prompt generator object based on the configuration.
    """

    __generator_map = {
        "splice": SplicePromptGenerator,
    }

    @staticmethod
    def create(config: Config) -> PromptGenerator:
        """
        Create the prompt generator object.

        Args:
            config (Config): The configuration object for the RAG pipeline.

        Raises:
            ValueError: If the prompt generator name is not found in the generator map.

        Returns:
            PromptGenerator: The prompt generator object.
        """
        prompt_generator_name = config.prompt.prompt_generator_name
        if prompt_generator_name not in PromptGeneratorFactory.__generator_map:
            raise ValueError(
                f"No prompt generator available for model: {prompt_generator_name}"
            )
        return PromptGeneratorFactory.__generator_map[prompt_generator_name](config)
