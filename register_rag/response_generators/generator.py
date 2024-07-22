from abc import ABC, abstractmethod
from typing import Dict, Any, List
import asyncio

from .response_message import ResponseMessage


class Generator(ABC):
    """
    Base class for all response generators.

    The generate method is the main method that should be called to
    generate a response. It should be implemented by the subclass.
    """

    def __init__(self) -> None:
        super().__init__()
        self.load_task: asyncio.Task = None

    @abstractmethod
    async def generate(
        self, prompt: str, history_messages: List[ResponseMessage] = None
    ) -> str:
        """
        Generate a response based on the prompt and history messages.

        Args:
            prompt (str): The prompt for which the response should be generated.
            history_messages (List[ResponseMessage]): The history of messages in the
            conversation. Should contain message and role for each message.

        Raises:
            NotImplementedError: This method should be implemented by the subclass.

        Returns:
            str: The generated response.
        """
        raise NotImplementedError(
            "The generate method must be implemented by the subclass."
        )

    async def message_merge(
        self,
        prompt: str,
        history_messages: List[ResponseMessage],
        system_prompt: str,
    ) -> List[ResponseMessage]:
        """
        Merge the prompt with the history messages.

        Args:
            prompt (str): The prompt for which the response should be generated.
            history_messages (List[ResponseMessage]): The history of messages in the
            conversation. Should contain message and role for each message.

        Returns:
            List[ResponseMessage]: The merged messages.
        """
        messages = []
        if system_prompt is not None:
            messages.append(ResponseMessage(message=system_prompt, role="system"))
        if history_messages is not None:
            for message in history_messages:
                messages.append(
                    ResponseMessage(message=message.message, role=message.role)
                )
        messages.append(ResponseMessage(message=prompt, role="user"))
        return messages
