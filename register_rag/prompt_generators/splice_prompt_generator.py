from typing import List

from register_rag.config.config import Config

from ..documents import Document
from .prompt_generator import PromptGenerator


class SplicePromptGenerator(PromptGenerator):
    """
    A prompt generator that generates a prompt by splicing the query and the page content of the related
    documents.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    async def get_prompt(self, query: str, related_docs: List[Document]) -> str:
        """
        Generate a prompt by splicing the query and the page content of the related documents.

        Args:
            query (str): The query to generate the prompt.
            related_docs (List[Document]): The related documents to generate the prompt.

        Returns:
            str: The generated prompt.
        """
        prompt = "请根据下面的信息：\n"
        for i, doc in enumerate(related_docs):
            prompt += f"{i + 1}. {doc.page_content}\n"
        prompt += f"回答问题：{query}"
        return prompt
