from typing import List


from ...config.config import Config
from ...documents import Document
from .remote_embedding_getter import RemoteEmbeddingGetter


class XinferenceEmbeddingGetter(RemoteEmbeddingGetter):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        try:
            import xinference_client
        except ImportError:
            raise ImportError(
                "xinference_client is required for xinference embedding getter"
                "Please install it using `pip install xinference_client`"
            )
        self.client = xinference_client.RESTfulClient(self.remote_url)
        try:
            self.model = self.client.get_model(self.model_name)
        except:
            try:
                self.client.launch_model(self.model_name, model_type="embedding")
            except:
                raise RuntimeError(f"Could not find or launch model {self.model_name}")
            self.model = self.client.get_model(self.model_name)

    async def embedding(self, docs: List[Document]) -> List[List[float]]:
        return [
            item["embedding"]
            for item in self.model.create_embedding([doc.page_content for doc in docs])[
                "data"
            ]
        ]

    async def load(self) -> None:
        pass
