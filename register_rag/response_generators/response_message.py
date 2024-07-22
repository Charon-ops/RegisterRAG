from typing import Literal
from pydantic import BaseModel


class ResponseMessage(BaseModel):
    """
    The message and role of a response.

    The role can be one of "system", "user", or "assistant". And the message
    is the actual text of the message
    """

    message: str
    role: Literal["system", "user", "assistant"]
