from typing import List
from register_rag.response_generators.response_message import ResponseMessage
from .generator import Generator


class OllamaGenerator(Generator):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name

    async def generate(
        self,
        prompt: str,
        history_messages: List[ResponseMessage] = None,
        system_prompt: str = None,
    ) -> str:
        try:
            import ollama
        except:
            raise ImportError(
                "The Ollama library is not installed. Please install it using `pip install ollama`."
            )

        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "message": system_prompt})

        if history_messages:
            for message in history_messages:
                messages.append({"role": message.role, "message": message.message})

        messages.append({"role": "user", "message": prompt})

        response = ollama.chat(
            model=self.model_name,
            messages=messages,
        )

        return response["message"]["content"]
