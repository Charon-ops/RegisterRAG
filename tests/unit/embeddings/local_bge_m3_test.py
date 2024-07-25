import sys
import os
import asyncio

sys.path.append(f"{sys.path[0]}/../../..")

from register_rag.embeddings.local import SentenceTransformerEmbeddingGetter
from register_rag.documents import Document
from register_rag.config import Config
from register_rag.config.embedding_config import EmbeddingConfig


def test_bge_m3_embedding_getter():
    """
    Test: BgeM3EmbeddingGetter
    """
    config_file = os.path.join(os.path.dirname(__file__), "local_bge_m3_test.json")
    config = Config.load(config_file)
    bge_m3_embedding_getter = SentenceTransformerEmbeddingGetter(config)
    docs = [
        Document(page_content="This is a test document.", metadata={"source": "test"})
    ]
    embeddings = asyncio.run(bge_m3_embedding_getter.get_embedding(docs))
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 1024
