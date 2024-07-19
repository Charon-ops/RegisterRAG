import sys
import asyncio
import os

sys.path.append(f"{sys.path[0]}/../../..")

from register_rag.documents.loader import TextLoader

# 这一行是为了保证推测出的编码是utf-8


def test_text_loader_directory():
    """
    Test: TextLoader
    """
    loader = TextLoader(os.path.dirname(__file__))

    res = asyncio.run(loader.load())

    file_list = os.listdir(os.path.dirname(__file__))
    file_list = [f for f in file_list if f.endswith(".py")]

    assert len(res) == len(file_list)

    for r in res:
        with open(r.metadata["src"], "r", encoding="UTF-8") as f:
            content = f.read()

        assert r.page_content == content
        assert r.metadata["encoding"] == "utf-8"
