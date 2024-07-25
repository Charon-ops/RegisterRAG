from abc import ABC, abstractmethod
from typing import List

from ..documents import Document
from ..config import Config


class PromptGenerator(ABC):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = Config

    @abstractmethod
    async def get_prompt(self, query: str, related_docs: List[Document]) -> str:
        """A method to generate a prompt for the query and related documents.

        Args:
            query (str): The query to generate the prompt.
            related_docs (List[Document]): The related documents to generate the prompt.

        Returns:
            str: The generated prompt.
        """
        raise NotImplementedError(
            "get_prompt method is not implemented, it should be implemented in the sub class."
        )
