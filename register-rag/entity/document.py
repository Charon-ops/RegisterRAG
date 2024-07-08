from typing import Dict
from pydantic import BaseModel


class Document(BaseModel):
    page_content: str
    metadata: Dict[str, str] = {}
