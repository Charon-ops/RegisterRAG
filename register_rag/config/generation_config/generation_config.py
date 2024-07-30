from typing import Optional, Literal
from pydantic import BaseModel

from .xinference_config import XinferenceConfig


class GenerationConfig(BaseModel):
    """
    The configuration class for the generation model.

    The configuration class contains the following attributes:

    - generation_type: Literal["local", "remote"]
    The type of the generation model. If "local", the model is loaded from the local file system.
    If "remote", the model is loaded from a remote URL. It can be an api endpoint, a docker container,
    or a remote server.

    - generation_model_name_or_path: Optional[str]
    The path and the detailed class name of the generation model. For example,
    "transformers/shenzhi-wang/Llama3.1-8B-Chinese-Chat".
    The first part of the string is the model name, and the second part is the model name or path. The
    second part can be a model name or a path to the model.

    - generation_model_device: str, default "cpu"
    The device on which the model is loaded. Default is "cpu". Only support pytorch models.

    - generation_model_preload: bool, default False
    Whether to preload the model. Default is False.

    - generation_remote_url: Optional[str]
    The URL of the remote generation model. Only used when generation_type is "remote".

    - generation_remote_token: Optional[str]
    The token for the remote generation model. Only used when generation_type is "remote". It is not
    required if the remote model does not need a token.

    - generation_xinference_config: Optional[XinferenceConfig]
    The configuration for the xinference model. Only used when generation_type is "remote", and the
    generator is "xinference".
    """

    generation_type: Literal["local", "remote"]
    generation_model_name_or_path: Optional[str] = None
    generation_model_preload: bool = False
    generation_model_device: str = "cpu"
    generation_remote_url: Optional[str] = None
    generation_remote_token: Optional[str] = None
    generation_xinference_config: Optional[XinferenceConfig] = None
