from typing import Optional
from pydantic import BaseModel


class StoreConfig(BaseModel):
    store_name: str
    store_remote_url: Optional[str] = None
    store_remore_token: Optional[str] = None
    store_local_path: Optional[str] = None
