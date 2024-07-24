import sys
import os
import asyncio

sys.path.append(f"{sys.path[0]}/../../..")

from register_rag.config import Config
from register_rag.response_generators import ResponseMessage
from register_rag.response_generators.remote import OllamaGenerator


def test_ollama_generator():
    config_path = os.path.join(os.path.dirname(__file__), "ollama_generator_test.json")
    config = Config.load(config_path)
    generator = OllamaGenerator(config)
    response = asyncio.run(
        generator.generate(
            prompt="Hello, how are you?", system_prompt="你需要以中文回答所有问题"
        )
    )
    assert response and len(response) > 0

    test_history = [
        ResponseMessage(content="你需要以中文回答所有问题", role="system"),
        ResponseMessage(content="How are you?", role="user"),
        ResponseMessage(content="I am good.", role="assistant"),
    ]

    response = asyncio.run(
        generator.generate("Hello, how are you?", history_messages=test_history)
    )
    assert response and len(response) > 0
