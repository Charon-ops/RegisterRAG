from typing import List, TypedDict, Literal

from ...config import Config
from ..response_message import ResponseMessage
from .remote_generator import RemoteGenerator


class OllamaGenerator(RemoteGenerator):
    """
    Ollama generator class.

    The Ollama generator uses the Ollama library to generate responses. You should
    run `ollama serve` to start the Ollama server before using this generator. The
    `ollama` library for Python is also required. You can install it using
    `pip install ollama`.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.model_name = "/".join(self.model_name.split("/")[1:])

    async def generate(
        self,
        prompt: str,
        history_messages: List[ResponseMessage] = None,
        system_prompt: str = None,
    ) -> str:
        """
        Generate a response using the Ollama library.

        Args:
            prompt (str): The prompt to generate the response.
            history_messages (List[ResponseMessage], optional): The history messages to generate the response.
            Defaults to None.
            system_prompt (str, optional): The system prompt to generate the response. Defaults to None.

        Raises:
            ImportError: If the Ollama library is not installed.

        Returns:
            str: The generated response.
        """
        try:
            import ollama
        except:
            raise ImportError(
                "The Ollama library is not installed. Please install it using `pip install ollama`."
            )

        messages = await self.message_merge(prompt, history_messages, system_prompt)

        response = ollama.chat(model=self.model_name, messages=messages)

        return response["message"]["content"]
