from pydantic import BaseModel


class EvaluateConfig(BaseModel):
    query_json_path: str
    response_json_path: str


class RecallEvaluateConfig(EvaluateConfig):
    store_route_path: str
    store_port: str
    embedding_route_path: str
    embedding_port: str
    max_tries: int
    save_path: str


class OurEvaluateConfig(EvaluateConfig):
    app_name: str
    store_name: str
    recall_json_path: str
    retrieve_top_k: int
    rerank_top_k: int
    save_path: str
