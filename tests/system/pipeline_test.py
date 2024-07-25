import sys
import os
import asyncio
import json

sys.path.append(f"{sys.path[0]}/../..")

from register_rag import Pipeline
from register_rag.documents.loader import PDFLoader
from register_rag.documents.splitter import CharacterSplitter


def test_pipeline():
    config_path = os.path.join(os.path.dirname(__file__), "pipeline_test.json")
    pipeline = Pipeline(config_path)

    loader = PDFLoader(os.path.join(os.path.dirname(__file__), "test.pdf"))
    docs = asyncio.run(loader.load_and_split(CharacterSplitter()))

    assert docs and len(docs) > 0

    asyncio.run(pipeline.add_docs(docs, collection_name="test"))

    query = "What are the main changes in architecture?"
    response, related_docs = asyncio.run(
        pipeline.get_response(
            query, collection_name="test", top_k=20, return_related_docs=True
        )
    )

    assert response and len(response) > 0

    output_json = {
        "query": query,
        "related_docs": [doc.page_content for doc in related_docs],
        "response": response,
    }
    with open(
        os.path.join(os.path.dirname(__file__), "pipeline_test_output.json"),
        "w",
        encoding="UTF-8",
    ) as f:
        json.dump(output_json, f, indent=4)
