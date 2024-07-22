import sys
import asyncio

sys.path.append(f"{sys.path[0]}/../../..")

from register_rag.response_generators.local import TransformersGenerator


def test_transformers_generator():
    generator = TransformersGenerator("Qwen/Qwen2-0.5B")
    response = asyncio.run(generator.generate("Hello, how are you?"))
    assert response
