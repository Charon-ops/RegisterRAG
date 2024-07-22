from typing import Literal
from pydantic import BaseModel


class ResponseMessage(BaseModel):
    message: str
    role: Literal["system", "user", "assistant"]
