import sys
import os

from langchain_core.documents import Document

sys.path.append(f"{sys.path[0]}/..")

from store.chroma_store import ChromaStore, ChromaStoreEmbeddingFunction

chroma_store = ChromaStore(os.path.join(os.path.dirname(__file__), "chroma"))

test_str = ["苹果是一种水果", "计算机病毒是一种软件", "橡胶是一种材料"]
test_documents = [
    Document(page_content=str, metadata={"character-length": len(str)})
    for str in test_str
]

chroma_store.add_documents(test_documents)

test_query = "什么是计算机病毒"

embedding_model = ChromaStoreEmbeddingFunction()
query_embedding = embedding_model([test_query])[0]

res = chroma_store.search_by_embed(query_embed=query_embedding, results=1)

print(res)
