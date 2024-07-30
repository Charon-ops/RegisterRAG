from pydantic import BaseModel


class XinferenceConfig(BaseModel):
    """
    The configuration class for the xinference model. It is only used when the generator is "xinference".

    The configuration class contains the following attributes:

    - xinference_model_engine: str
    The engine of the xinference model. Default is "Transformers".

    - xinference_model_format: str
    The format of the xinference model. Default is "pytorch".

    - xinference_ngpu: str
    The number of GPUs to use. Default is "auto".

    - xinference_model_size: str
    The size of the xinference model. For example, "1_5, "7", "72"

    - xinference_model_quantization: str
    The quantization of the xinference model. For example, "4-bit", "None"
    """

    xinference_model_engine: str = "Transformers"
    xinference_model_format: str = "pytorch"
    xinference_ngpu: str = "auto"
    xinference_model_size: str
    xinference_mdoel_quantization: str
