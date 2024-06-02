from typing import Union
from fastapi import UploadFile
from pydantic import BaseModel


class InsertDocParams(BaseModel):
    file_content: Union[str, UploadFile]
    file_type: str
    app_name: str
    begin_index: int = 0
    doc_name: str = None
    doc_id: int = None
