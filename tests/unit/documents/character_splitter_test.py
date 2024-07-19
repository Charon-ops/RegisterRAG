import sys
import asyncio

sys.path.append(f"{sys.path[0]}/../../..")

from register_rag.documents.splitter import CharacterSplitter
from register_rag.documents import Document

# 这一行是为了保证推测出的编码是utf-8


def test_character_splitter():
    splitter = CharacterSplitter()

    test_metadata = {"src": "test"}
    test_doc = Document(
        page_content="This is the first line.\nThis is the second line.",
        metadata=test_metadata,
    )

    res = asyncio.run(
        splitter.split(test_doc, separator="\n", metadata={"key": "value"})
    )

    assert len(res) == 2
    assert res[0].page_content == "This is the first line."
    assert res[0].metadata["src"] == "test"
    assert res[0].metadata["key"] == "value"
