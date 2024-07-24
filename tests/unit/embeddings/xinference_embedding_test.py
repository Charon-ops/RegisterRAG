import sys
import os
import asyncio

sys.path.append(f"{sys.path[0]}/../../..")

from register_rag.config import Config
from register_rag.embeddings.remote import XinferenceEmbeddingGetter
from register_rag.documents import Document


def test_xinference_embedding_getter():
    config_path = os.path.join(
        os.path.dirname(__file__), "xinference_embedding_test.json"
    )
    config = Config.load(config_path)
    embedding_getter = XinferenceEmbeddingGetter(config)
    test_docs = [
        Document(page_content="This is a test document.", metadata={"source": "test"}),
        Document(
            page_content="This is another test document.", metadata={"source": "test"}
        ),
    ]
    embeddings = asyncio.run(embedding_getter.embedding(test_docs))
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 1024
