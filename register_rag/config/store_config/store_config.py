from typing import Optional, Literal
from pydantic import BaseModel


class StoreConfig(BaseModel):
    store_type: Literal["local", "remote"]
    store_name: Optional[str] = None
    store_remote_url: Optional[str] = None
    store_remote_token: Optional[str] = None
    store_local_path: Optional[str] = None
