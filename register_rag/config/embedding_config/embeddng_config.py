from typing import Optional, Literal
from pydantic import BaseModel


class EmbeddingConfig(BaseModel):
    """
    The configuration class for the embedding model.

    The configuration class contains the following attributes:

    - embedding_type: Literal["local", "remote"]
    The type of the embedding model. If "local", the model is loaded from the local file system.
    If "remote", the model is loaded from a remote URL. It can be an api endpoint, a docker container,
    or a remote server.

    - embedding_model_name_or_path: Optional[str]
    The path and the detailed class name of the embedding model. For example, "sentence_transformer/BAAI/bge-m3".
    The first part of the string is the model name, and the second part is the model name or path.

    - embedding_model_device: str
    The device on which the model is loaded. Default is "cpu". Only support pytorch models.

    - embedding_model_preload: bool
    Whether to preload the model. Default is False.

    - embedding_remote_url: Optional[str]
    The URL of the remote embedding model. Only used when embedding_type is "remote".

    - embedding_remote_token: Optional[str]
    The token for the remote embedding model. Only used when embedding_type is "remote". It is not
    required if the remote model does not need a token.
    """

    embedding_type: Literal["local", "remote"]
    embedding_model_name_or_path: Optional[str] = None
    embedding_model_device: str = "cpu"
    embedding_model_preload: bool = False
    embedding_remote_url: Optional[str] = None
    embedding_remote_token: Optional[str] = None
