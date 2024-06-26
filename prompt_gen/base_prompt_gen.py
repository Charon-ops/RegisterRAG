# import ollama
from typing import List
from langchain_core.documents import Document
from .base import PromptGen
import requests


class BasePromptGen(PromptGen):
    def __init__(self) -> None:
        super().__init__()
        return

    def prompt_gen(self, query, contents: List[Document]) -> str:
        prompt_merge = ""
        for cont in contents:
            prompt_merge += f"{cont.page_content}\n"
        prompt_merge += f"以上是有关于问题的参考资料，请根据上面的资料回答问题：{query}"
        return prompt_merge
