from pydantic import BaseModel


class GetResponseParams(BaseModel):
    query_content: str
    app_name: str
    config_path: str = "app_register_config.json"
