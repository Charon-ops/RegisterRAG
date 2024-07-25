from pydantic import BaseModel


class PromptConfig(BaseModel):
    prompt_generator_name: str = "splice"
