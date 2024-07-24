import sys
import os
import asyncio

sys.path.append(f"{sys.path[0]}/../../..")

from register_rag.config import Config
from register_rag.response_generators import ResponseMessage
from register_rag.response_generators.local import TransformersGenerator


def test_transformers_generator():
    config_path = os.path.join(
        os.path.dirname(__file__), "transformers_generator_test.json"
    )
    config = Config.load(config_path)
    generator = TransformersGenerator(config)
    response = asyncio.run(generator.generate("Hello, how are you?"))
    assert response and len(response) > 0

    test_history = [
        ResponseMessage(content="How are you?", role="user"),
        ResponseMessage(content="I am good.", role="assistant"),
    ]

    response = asyncio.run(
        generator.generate("Hello, how are you?", history_messages=test_history)
    )
    assert response and len(response) > 0
