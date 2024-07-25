from typing import List

from register_rag.config.config import Config

from ..documents import Document
from .prompt_generator import PromptGenerator


class SplicePromptGenerator(PromptGenerator):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

    async def get_prompt(self, query: str, related_docs: List[Document]) -> str:
        prompt = "请根据下面的信息：\n"
        for i, doc in enumerate(related_docs):
            prompt += f"{i + 1}. {doc.page_content}\n"
        prompt += f"回答问题：{query}"
        return prompt
