from ..config import Config
from ..prompt_generators import PromptGenerator, SplicePromptGenerator


class PromptGeneratorFactory:
    __generator_map = {
        "splice": SplicePromptGenerator,
    }

    @staticmethod
    def create(config: Config) -> PromptGenerator:
        prompt_generator_name = config.prompt.prompt_generator_name
        if prompt_generator_name not in PromptGeneratorFactory.__generator_map:
            raise ValueError(
                f"No prompt generator available for model: {prompt_generator_name}"
            )
        return PromptGeneratorFactory.__generator_map[prompt_generator_name](config)
