from typing import List
import gradio as gr

from .embeddings import local as local_embedding
from .embeddings import remote as remote_embedding
from .store import local as local_store
from .response_generators import local as local_response_generator
from .response_generators import remote as remote_response_generator


class Dashboard:
    def __init__(self) -> None:
        self.__embedding_models = [
            "local/" + model for model in local_embedding.__all__
        ] + ["remote/" + model for model in (remote_embedding.__all__)]
        self.__embedding_models.remove("local/LocalEmbeddingGetter")
        self.__embedding_models.remove("remote/RemoteEmbeddingGetter")

        self.__store_models = ["local/" + model for model in local_store.__all__]
        self.__store_models.remove("local/LocalStore")

        self.__response_generators = [
            "local/" + model for model in local_response_generator.__all__
        ] + ["remote/" + model for model in remote_response_generator.__all__]
        self.__response_generators.remove("local/LocalGenerator")
        self.__response_generators.remove("remote/RemoteGenerator")

        self.app = self.init_app()

    def init_app(self) -> gr.Blocks:
        with gr.Blocks() as app:
            with gr.Tab("Upload"):
                with gr.Row():
                    with gr.Column():
                        upload_data_label = gr.Label(
                            label="Upload configuration", value="System configuration"
                        )
                        upload_config_file = gr.File(
                            label="Upload config file", file_types=["json"]
                        )

                with gr.Row():
                    with gr.Column():
                        upload_embedding_label = gr.Label(
                            label="Embedding model configuration",
                            value="Embedding model configuration",
                        )
                        with gr.Row():
                            upload_embedding_model = gr.Dropdown(
                                label="Select embedding model",
                                choices=self.__embedding_models,
                            )
                            upload_embedding_model_name_or_path = gr.Textbox(
                                label="Model name or path",
                                placeholder="e.g. 'BAAI/bge-m3'",
                            )
                            upload_embedding_model_device = gr.Radio(
                                label="Device",
                                choices=["cpu", "cuda"],
                            )
                            upload_embedding_model_preload = gr.Checkbox(
                                label="Preload model",
                            )

                        with gr.Row():
                            upload_embedding_remote_url = gr.Textbox(
                                label="Embedding Remote URL",
                                placeholder="e.g. 'http://localhost:9997'",
                            )
                            upload_embedding_remote_token = gr.Textbox(
                                label="Embedding Remote token",
                                placeholder="your token or api-key here",
                            )

                    with gr.Column():
                        upload_store_label = gr.Label(
                            label="Store configuration", value="Store configuration"
                        )
                        with gr.Row():
                            upload_store_model = gr.Dropdown(
                                label="Select store model", choices=self.__store_models
                            )
                            upload_store_local_path = gr.Textbox(
                                label="Local path",
                                placeholder="e.g. '/path/to/store'",
                            )

                        with gr.Row():
                            upload_store_remote_url = gr.Textbox(
                                label="Store Remote URL",
                                placeholder="e.g. 'http://localhost:9997'",
                            )
                            upload_store_remote_token = gr.Textbox(
                                label="Store Remote token",
                                placeholder="your token or api-key here",
                            )

                with gr.Row():
                    with gr.Column():
                        upload_files_label = gr.Label(
                            label="Upload files", value="Upload files"
                        )
                        upload_files = gr.File(
                            label="Upload files",
                            file_count="multiple",
                        )

                    with gr.Column():
                        upload_label = gr.Label(label="Upload", value="Upload")
                        save_config_path = gr.Textbox(
                            label="Save configuration to",
                            placeholder="e.g. '/path/to/save'",
                        )
                        save_config_path_button = gr.Button(value="Save")
                        upload_button = gr.Button(value="Upload")
                        upload_res = gr.Text(label="Upload result", value="")

            with gr.Tab("Response"):
                with gr.Row():
                    with gr.Column():
                        response_config_label = gr.Label(
                            label="Response configuration",
                            value="Response configuration",
                        )
                        response_config_file = gr.File(
                            label="Upload config file",
                            file_types=["json"],
                        )

                with gr.Row():
                    with gr.Column():
                        resposne_embedding_model_label = gr.Label(
                            label="Embedding model configuration",
                            value="Embedding model configuration",
                        )
                        with gr.Row():
                            response_embedding_model = gr.Dropdown(
                                label="Select embedding model",
                                choices=self.__embedding_models,
                            )
                            response_embedding_model_name_or_path = gr.Textbox(
                                label="Model name or path",
                                placeholder="e.g. 'BAAI/bge-m3'",
                            )
                            response_embedding_model_device = gr.Radio(
                                label="Device",
                                choices=["cpu", "cuda"],
                            )
                            response_embedding_model_preload = gr.Checkbox(
                                label="Preload model",
                            )

                        with gr.Row():
                            response_embedding_remote_url = gr.Textbox(
                                label="Embedding Remote URL",
                                placeholder="e.g. 'http://localhost:9997'",
                            )
                            response_embedding_remote_token = gr.Textbox(
                                label="Embedding Remote token",
                                placeholder="your token or api-key here",
                            )

                    with gr.Column():
                        response_store_label = gr.Label(
                            label="Store configuration", value="Store configuration"
                        )
                        with gr.Row():
                            response_store_model = gr.Dropdown(
                                label="Select store model", choices=self.__store_models
                            )
                            response_store_local_path = gr.Textbox(
                                label="Local path",
                                placeholder="e.g. '/path/to/store'",
                            )

                        with gr.Row():
                            response_store_remote_url = gr.Textbox(
                                label="Store Remote URL",
                                placeholder="e.g. 'http://localhost:9997'",
                            )
                            response_store_remote_token = gr.Textbox(
                                label="Store Remote token",
                                placeholder="your token or api-key here",
                            )

                with gr.Row():
                    with gr.Row():
                        response_generator_label = gr.Label(
                            label="Response generator configuration",
                            value="Response generator configuration",
                        )

                with gr.Row():
                    with gr.Column():
                        response_generator = gr.Dropdown(
                            label="Select response generator",
                            choices=self.__response_generators,
                        )
                        response_generator_name_or_path = gr.Textbox(
                            label="Model name or path",
                        )
                        response_generator_device = gr.Radio(
                            label="Device",
                            choices=["cpu", "cuda"],
                        )
                        response_generator_preload = gr.Checkbox(label="Preload model")
                        response_generator_remote_url = gr.Textbox(
                            label="Response generator Remote URL",
                            placeholder="e.g. 'http://localhost:9997'",
                        )
                        response_generator_remote_token = gr.Textbox(
                            label="Response generator Remote token",
                            placeholder="your token or api-key here",
                        )

                    with gr.Column():
                        gr.Label(label="Xinference", value="Xinference Configuration")
                        xinference_model_engine = gr.Dropdown(
                            label="Model engine",
                            choices=["Transformers", "vLLM", "llama.cpp"],
                        )
                        xinference_model_format = gr.Dropdown(
                            label="Model format",
                            choices=["pytorch", "gptq", "awq", "ggufv2"],
                        )
                        xinference_model_size = (
                            gr.Textbox(
                                label="Model size",
                                placeholder="e.g. '1_5', '72'",
                            ),
                        )
                        xinference_model_quantization = (
                            gr.Textbox(
                                label="Model quantization",
                                placeholder="e.g. '4-bit', '8-bit'",
                            ),
                        )
                        xinference_ngpu = (
                            gr.Textbox(
                                label="Number of GPUs",
                                placeholder="e.g. 'auto', '1', 'CPU'",
                            ),
                        )

                with gr.Row():
                    gr.Label(label="Upload & Response", value="Upload data")

                with gr.Row():
                    with gr.Column():
                        response_file = gr.File(
                            label="Upload files",
                        )

                    with gr.Column():
                        response_button = gr.Button(value="Response")
                        response_res = gr.Text(label="Response result", value="")

            return app

    def launch(self):
        self.app.launch()
