import sys
import asyncio

sys.path.append(f"{sys.path[0]}/../../..")

from register_rag.response_generators import ResponseMessage
from register_rag.response_generators.remote import OllamaGenerator


def test_ollama_generator():
    generator = OllamaGenerator("qwen2:1.5b")
    response = asyncio.run(
        generator.generate(
            "Hello, how are you?", system_prompt="你需要以中文回答所有问题"
        )
    )
    assert response

    test_history = [
        ResponseMessage(message="你需要以中文回答所有问题", role="system"),
        ResponseMessage(message="How are you?", role="user"),
        ResponseMessage(message="I am good.", role="assistant"),
    ]

    response = asyncio.run(
        generator.generate("Hello, how are you?", history_messages=test_history)
    )
    assert response
