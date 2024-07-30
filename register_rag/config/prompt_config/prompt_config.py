from pydantic import BaseModel


class PromptConfig(BaseModel):
    """
    The configuration class for the prompt model.

    The configuration class contains the following attributes:

    - prompt_generator_name: str, default "splice"
    The name of the prompt generator. Default is "splice".
    The default template is:

    ```plaintext
    请根据以下信息:
    {recall_res}
    回答问题:{query}
    ```

    where {recall_res} is the retrieved document content and {query} is the query.
    """

    prompt_generator_name: str = "splice"
