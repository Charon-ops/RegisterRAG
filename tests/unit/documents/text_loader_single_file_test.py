import sys
import asyncio
import os

sys.path.append(f"{sys.path[0]}/../../..")

from register_rag.documents.loader import TextLoader

# 这一行是为了保证推测出的编码是utf-8


def test_text_loader_single_file():
    """
    Test: TextLoader
    """
    loader = TextLoader(os.path.abspath(__file__))

    res = asyncio.run(loader.load())

    assert len(res) == 1

    with open(os.path.abspath(__file__), "r") as f:
        content = f.read()

    assert res[0].page_content == content
    assert res[0].metadata["src"] == os.path.abspath(__file__)
    assert res[0].metadata["encoding"] == "utf-8"
