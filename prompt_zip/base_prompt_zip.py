# import ollama
from typing import List
from langchain_core.documents import Document
from .base import PromptZip
import requests


class BasePromptZip(PromptZip):
    def __init__(self, router_path: str, port: str) -> None:
        super().__init__()
        self.router_path = router_path
        self.port = port
        self.request_url = f"{self.router_path}:{self.port}"
        return

    def prompt_zip(
        self,
        prompt: List[Document],
        query: str,
        rate: float = 0.5,
        instruction: str = "",
    ) -> str:
        return "\n".join([doc.page_content for doc in prompt])
