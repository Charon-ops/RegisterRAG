from typing import Optional, Literal
from pydantic import BaseModel


class EmbeddingConfig(BaseModel):
    embedding_type: Literal["local", "remote"]
    embedding_model_name_or_path: Optional[str] = None
    embedding_model_device: str = "cpu"
    embedding_model_preload: bool = False
    embedding_remote_url: Optional[str] = None
    embedding_remote_token: Optional[str] = None
