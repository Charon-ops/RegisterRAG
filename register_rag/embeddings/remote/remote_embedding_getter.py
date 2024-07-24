from ...config.config import Config
from ..embedding_getter import EmbeddingGetter


class RemoteEmbeddingGetter(EmbeddingGetter):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if self.config.embedding.embedding_model_name_or_path is None:
            raise ValueError(
                "embedding_model_name_or_path is required for remote embedding getter"
            )
        if self.config.embedding.embedding_remote_url is None:
            raise ValueError(
                "embedding_remote_url is required for remote embedding getter"
            )
        self.model_name = self.config.embedding.embedding_model_name_or_path
        self.model_name = "/".join(self.model_name.split("/")[1:])
        self.remote_url = self.config.embedding.embedding_remote_url
        self.token = self.config.embedding.embedding_remote_token
