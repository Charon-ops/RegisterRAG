from typing import Optional, Literal
from pydantic import BaseModel

from .xinference_config import XinferenceConfig


class GenerationConfig(BaseModel):
    generation_type: Literal["local", "remote"]
    generation_model_name_or_path: Optional[str] = None
    generation_model_preload: bool = False
    generation_model_device: str = "cpu"
    generation_remote_url: Optional[str] = None
    generation_remote_token: Optional[str] = None
    generation_xinference_config: Optional[XinferenceConfig] = None
