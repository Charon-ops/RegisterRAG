from abc import abstractmethod
from typing import List

from ...config import Config
from ..response_message import ResponseMessage

from .. import Generator


class RemoteGenerator(Generator):
    """
    Remote generator base class.

    The model name is required for the remote generator.

    The generate method should be implemented by the subclass.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if self.config.generation.generation_model_name_or_path is None:
            raise ValueError("Model name is required for remote generators.")
        if self.config.generation.generation_remote_url is None:
            raise ValueError("Remote URL is required for remote generators.")
        self.model_name = self.config.generation.generation_model_name_or_path
        self.remote_url = self.config.generation.generation_remote_url
        self.token = self.config.generation.generation_remote_token

    async def generate(
        self,
        prompt: str,
        history_messages: List[ResponseMessage] = None,
        system_prompt: str = None,
    ) -> str:
        raise NotImplementedError(
            "The generate method must be implemented by the subclass."
        )
