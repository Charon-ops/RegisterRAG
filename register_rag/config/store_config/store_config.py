from typing import Optional, Literal
from pydantic import BaseModel


class StoreConfig(BaseModel):
    """
    The configuration class for the store model.

    The configuration class contains the following attributes:

    - store_type: Literal["local", "remote"]
    The type of the store model. If "local", the model is loaded from the local file system.
    If "remote", the model is loaded from a remote URL. It can be an api endpoint, a docker container,
    or a remote server.

    - store_name: Optional[str]
    The name of the store model. For example, "chroma".

    - store_remote_url: Optional[str]
    The URL of the remote store model. Only used when store_type is "remote".

    - store_remote_token: Optional[str]
    The token for the remote store model. Only used when store_type is "remote". It is not
    required if the remote model does not need a token.

    - store_local_path: Optional[str]
    The path of the local store model. Only used when store_type is "local".
    """

    store_type: Literal["local", "remote"]
    store_name: Optional[str] = None
    store_remote_url: Optional[str] = None
    store_remote_token: Optional[str] = None
    store_local_path: Optional[str] = None
