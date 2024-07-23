from pydantic import BaseModel

from .store_config import StoreConfig


class Config(BaseModel):
    store: StoreConfig
