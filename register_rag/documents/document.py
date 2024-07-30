from typing import Dict, Any
from pydantic import BaseModel


class Document(BaseModel):
    """
    A document class that represents a single document. The document contains the page content and
    metadata. The metadata is a dictionary that contains additional information about the document.
    """

    page_content: str
    metadata: Dict[str, Any] | None = None
