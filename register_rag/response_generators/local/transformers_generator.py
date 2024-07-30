import asyncio
from typing import List, Dict

from transformers import pipeline
import torch

from ...config import Config
from .. import ResponseMessage
from .local_generator import LocalGenerator


class TransformersGenerator(LocalGenerator):
    """
    Transformers generator class.

    The generator will use GPU if available.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.model_path = "/".join(self.model_path.split("/")[1:])

    async def load(self) -> None:
        """
        Load the model.

        The model will be created using the pipeline from the transformers library.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = pipeline(
            "text-generation",
            model=self.model_path,
            tokenizer=self.model_path,
            device=device,
        )

    async def generate(
        self,
        prompt: str,
        history_messages: List[ResponseMessage] = None,
        system_prompt: str = None,
    ) -> str:
        """
        Generate a response based on the prompt and history messages.

        Args:
            prompt (str): The prompt for which the response should be generated.
            history_messages (List[ResponseMessage], optional): The history of messages in the
            conversation. Should contain message and role for each message. Defaults to None.
            system_prompt (str, optional): The system prompt for the conversation.
            Defaults to None.

        Returns:
            str: The generated response.
        """
        if self.load_task is None:
            self.load_task = asyncio.create_task(self.load())
        await self.load_task
        messages = await self.message_merge(prompt, history_messages, system_prompt)
        response = self.model(messages)
        return response[0]["generated_text"][-1]["content"]