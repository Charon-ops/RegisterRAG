import sys
import asyncio

sys.path.append(f"{sys.path[0]}/../../..")

from register_rag.embeddings.local import BgeM3EmbeddingGetter
from register_rag.documents import Document


def test_bge_m3_embedding_getter():
    """
    Test: BgeM3EmbeddingGetter
    """
    weight_path = "/home/yumuzhihan/Documents/Code/Project/RegisterRAG/data/bge-m3"
    bge_m3_embedding_getter = BgeM3EmbeddingGetter(
        weight_path=weight_path, device="cpu"
    )
    docs = [
        Document(page_content="This is a test document.", metadata={"source": "test"})
    ]
    embeddings = asyncio.run(bge_m3_embedding_getter.get_embedding(docs))
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 1024
