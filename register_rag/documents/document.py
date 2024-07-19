from typing import Dict, Any
from pydantic import BaseModel


class Document(BaseModel):
    page_content: str
    metadata: Dict[str, Any] = {}
