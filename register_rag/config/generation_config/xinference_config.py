from pydantic import BaseModel


class XinferenceConfig(BaseModel):
    xinference_model_engine: str = "Transformers"
    xinference_model_format: str = "pytorch"
    xinference_ngpu: str = "auto"
    xinference_model_size: str
    xinference_mdoel_quantization: str
