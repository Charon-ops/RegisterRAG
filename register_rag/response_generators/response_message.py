from typing import Literal, TypedDict


class ResponseMessage(TypedDict):
    """
    The message and role of a response.

    The role can be one of "system", "user", or "assistant". And the message
    is the actual text of the message
    """

    content: str
    role: Literal["system", "user", "assistant"]
