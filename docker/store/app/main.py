from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from EmbeddingStore.AnnStore import AnnoyStore


class DocEmbs(BaseModel):
    doc_list: List[str]
    doc_emb_list: List[List[float]]
    doc_index: List[int] = None
    doc_name: str = None
    doc_id: int = None
    store_name: str = "store"


class SearchParams(BaseModel):
    query_vec: List[float]
    num: int = 50
    store_name: str = "store"


class Docs(BaseModel):
    docs: List[str]
    store_name: str = "store"


class Ids(BaseModel):
    ids: List[int]
    store_name: str = "store"


app = FastAPI()


@app.post("/ann/add_docs")
def add_document(req: DocEmbs):
    doc_list = req.doc_list
    doc_emb_list = req.doc_emb_list
    doc_index = req.doc_index
    doc_name = req.doc_name
    doc_id = req.doc_id
    store_name = req.store_name
    store = AnnoyStore(store_name=store_name)
    store.add_documents(
        doc_list=doc_list,
        doc_emb_list=doc_emb_list,
        doc_index=doc_index,
        doc_name=doc_name,
        doc_id=doc_id,
    )


@app.post("/ann/search")
def search_by_embedding(req: SearchParams):
    query_vec = req.query_vec
    num = req.num
    store_name = req.store_name
    store = AnnoyStore(store_name=store_name)
    res = store.search_by_embedding(query_vec, num)
    return {"knowledges": res}


@app.post("/ann/get_ids")
def get_id_by_docs(req: Docs):
    docs = req.docs
    store_name = req.store_name
    store = AnnoyStore(store_name=store_name)
    ids = []
    for doc in docs:
        ids.append(store.get_id_by_doc(doc))
    return {"ids": ids}


@app.delete("/ann/remove_items")
def delete_by_ids(req: Ids):
    ids = req.ids
    store_name = req.store_name
    store = AnnoyStore(store_name=store_name)
    for id in ids:
        store.delete_by_id(id)
