import sys
import os
import asyncio


sys.path.append(f"{sys.path[0]}/../../..")

from register_rag.config import Config
from register_rag.response_generators.remote import XinferenceGenerator


def test_xinference_generator():
    config = Config.load(f"{os.path.dirname(__file__)}/xinference_generator_test.json")
    generator = XinferenceGenerator(config)
    response = asyncio.run(generator.generate("Hello, how are you?"))
    assert response and len(response) > 0
