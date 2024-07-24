import sys
import asyncio
import os

sys.path.append(f"{sys.path[0]}/../../..")

from register_rag.config import Config
from register_rag.documents import Document
from register_rag.store.local import ChromaStore
from register_rag.config.store_config import StoreConfig


def test_chroma_store():
    config_path = os.path.join(os.path.dirname(__file__), "chroma_store_test.json")
    config = Config.load(config_path)
    docs = [
        Document(page_content="This is a test document.", metadata={"source": "test"}),
        Document(
            page_content="This is another test document.", metadata={"source": "test"}
        ),
    ]
    store = ChromaStore(config)
    embeddings = [
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
    ]
    collection_name = "test"
    ids = ["1", "2"]
    asyncio.run(store.add_documents(docs, embeddings, collection_name, ids))
    test_get_id = asyncio.run(store.get_id_by_document(docs[0], collection_name))
    assert test_get_id == "1"
    test_search = asyncio.run(
        store.search_by_embedding([0.1, 0.2, 0.3], collection_name, top_k=1)
    )
    assert test_search[0].page_content == "This is a test document."
    asyncio.run(store.delete_by_id("1", collection_name))
    test_get_id = asyncio.run(store.get_id_by_document(docs[0], collection_name))
    assert test_get_id != "1"
