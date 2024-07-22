import asyncio
from typing import List, Dict

from transformers import pipeline
import torch

from .. import ResponseMessage
from .local_generator import LocalGenerator


class TransformersGenerator(LocalGenerator):
    def __init__(self, model_path: str, pre_load: bool = False) -> None:
        super().__init__(model_path, pre_load)
        self.tokenizer = None

    async def load(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = pipeline(
            "text-generation",
            model=self.model_path,
            tokenizer=self.model_path,
            device=device,
        )

    async def convert_messages_to_model_input(
        self, messages: List[ResponseMessage]
    ) -> List[Dict[str, str]]:
        return [
            {"role": message.role, "content": message.message} for message in messages
        ]

    async def generate(
        self,
        prompt: str,
        history_messages: List[ResponseMessage] = None,
        system_prompt: str = None,
    ) -> str:
        if self.load_task is None:
            self.load_task = asyncio.create_task(self.load())
        await self.load_task
        messages = await self.message_merge(prompt, history_messages, system_prompt)
        messages = await self.convert_messages_to_model_input(messages)
        response = self.model(messages)
        return response[0]["generated_text"][-1]["content"]
