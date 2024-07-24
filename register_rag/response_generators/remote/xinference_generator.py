from typing import List
from register_rag.config.config import Config
from register_rag.response_generators.response_message import ResponseMessage
from .remote_generator import RemoteGenerator


class XinferenceGenerator(RemoteGenerator):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        try:
            import xinference_client
        except ImportError:
            raise ImportError(
                "xinference_client is required for xinference generator"
                "Please install it using `pip install xinference_client`"
            )
        self.model_name = "/".join(self.model_name.split("/")[1:])
        self.client = xinference_client.RESTfulClient(self.remote_url)
        try:
            self.model = self.client.get_model(self.model_name)
        except:
            if self.config.generation.generation_xinference_config is None:
                raise ValueError(
                    "Xinference config is required for xinference generators to launch the model."
                )
            xinference_config = self.config.generation.generation_xinference_config
            try:
                self.client.launch_model(
                    self.model_name,
                    model_type="LLM",
                    model_engine=xinference_config.xinference_model_engine,
                    model_format=xinference_config.xinference_model_format,
                    model_size_in_billions=xinference_config.xinference_model_size,
                    quantization=xinference_config.xinference_mdoel_quantization,
                    n_gpu=xinference_config.xinference_ngpu,
                )
            except:
                raise RuntimeError(f"Could not find or launch model {self.model_name}")
            self.model = self.client.get_model(self.model_name)

    async def generate(
        self,
        prompt: str,
        history_messages: List[ResponseMessage] = None,
        system_prompt: str = None,
    ) -> str:
        if not history_messages:
            history_messages = []
        response = self.model.chat(
            prompt=prompt,
            chat_history=[message["content"] for message in history_messages],
            system_prompt=system_prompt,
        )
        return response["choices"][0]["message"]["content"]
