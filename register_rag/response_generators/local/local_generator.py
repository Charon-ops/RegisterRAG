from abc import abstractmethod
import asyncio
from typing import List

from register_rag.response_generators.response_message import ResponseMessage

from .. import Generator


class LocalGenerator(Generator):
    def __init__(self, model_path: str, pre_load: bool = False) -> None:
        super().__init__()
        self.model_path = model_path
        self.pre_load = pre_load
        self.load_task: asyncio.Task = None
        self.model = None
        if self.pre_load:
            self.load_task = asyncio.create_task(self.load())

    async def generate(
        self,
        prompt: str,
        history_messages: List[ResponseMessage] = None,
        system_prompt: str = None,
    ) -> str:
        if self.load_task is None:
            self.load_task = asyncio.create_task(self.load())
        await self.load_task
        return ""

    @abstractmethod
    async def load(self) -> None:
        raise NotImplementedError(
            "The load method must be implemented by the subclass."
        )
