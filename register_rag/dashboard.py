from typing import List, Tuple
import gradio as gr
import json

from . import Pipeline
from .config import Config
from .config.embedding_config import EmbeddingConfig
from .config.store_config import StoreConfig
from .config.generation_config import GenerationConfig, XinferenceConfig
from .module_factory import EmbeddingFactory, StoreFactory, ResponseGeneratorFactory
from .embeddings import MeanPoolingCalculator
from .embeddings import local as local_embedding
from .embeddings import remote as remote_embedding
from .store import local as local_store
from .response_generators import local as local_response_generator
from .response_generators import remote as remote_response_generator
from .documents import Document
from .documents.loader import TextLoader, PDFLoader, SqliteLoader
from .documents.splitter import CharacterSplitter


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
        self.pipeline = None

    def convert(self, value: Tuple) -> Tuple[str]:
        res: Tuple = ()
        for v in value:
            if isinstance(v, tuple):
                res.extend(self.convert_to_list(v))
            else:
                res += (v if v else "",)

        return res

    def set_config_from_file(self, config_file: str):
        config = Config.load(config_file)
        embedding_config = config.embedding
        store_config = config.store
        generation_config = config.generation
        xinference_config = generation_config.generation_xinference_config
        return self.convert(
            (
                embedding_config.embedding_type
                + "/"
                + EmbeddingFactory.get_class_name_from_config(config),
                "/".join(embedding_config.embedding_model_name_or_path.split("/")[1:]),
                embedding_config.embedding_model_device,
                embedding_config.embedding_model_preload,
                embedding_config.embedding_remote_url,
                embedding_config.embedding_remote_token,
                store_config.store_type
                + "/"
                + StoreFactory.get_class_name_from_config(config),
                store_config.store_local_path,
                store_config.store_remote_url,
                store_config.store_remote_token,
                generation_config.generation_type
                + "/"
                + ResponseGeneratorFactory.get_class_name_from_config(config),
                generation_config.generation_model_name_or_path,
                generation_config.generation_model_device,
                generation_config.generation_model_preload,
                generation_config.generation_remote_url,
                generation_config.generation_remote_token,
                xinference_config.xinference_model_engine,
                xinference_config.xinference_model_format,
                xinference_config.xinference_model_size,
                xinference_config.xinference_mdoel_quantization,
                xinference_config.xinference_ngpu,
            )
        )

    def save_config_from_ui(
        self,
        embedding_model: str,
        embedding_model_name_or_path: str,
        embedding_model_device: str,
        embedding_model_preload: bool,
        embedding_remote_url: str,
        embedding_remote_token: str,
        store_model: str,
        store_local_path: str,
        store_remote_url: str,
        store_remote_token: str,
        generator: str,
        generator_name_or_path: str,
        generator_device: str,
        generator_preload: bool,
        generator_remote_url: str,
        generator_remote_token: str,
        xinference_model_engine: str,
        xinference_model_format: str,
        xinference_model_size: str,
        xinference_model_quantization: str,
        xinference_ngpu: str,
        save_path: str,
    ) -> str:
        embedding_type = embedding_model.split("/")[0]
        embedding_model_name = (
            EmbeddingFactory.get_config_name_from_class_name(
                embedding_model.split("/")[1]
            )
            + "/"
            + embedding_model_name_or_path
        )
        store_type = store_model.split("/")[0]
        store_name = StoreFactory.get_config_name_from_class_name(
            store_model.split("/")[1]
        )
        generation_type = generator.split("/")[0]
        generation_name = (
            ResponseGeneratorFactory.get_config_name_from_class_name(
                generator.split("/")[1]
            )
            + "/"
            + generator_name_or_path
        )
        config = Config(
            embedding=EmbeddingConfig(
                embedding_type=embedding_type,
                embedding_model_name_or_path=embedding_model_name,
                embedding_model_device=embedding_model_device,
                embedding_model_preload=embedding_model_preload,
                embedding_remote_url=embedding_remote_url,
                embedding_remote_token=embedding_remote_token,
            ),
            store=StoreConfig(
                store_type=store_type,
                store_name=store_name,
                store_local_path=store_local_path,
                store_remote_url=store_remote_url,
                store_remote_token=store_remote_token,
            ),
            generation=GenerationConfig(
                generation_type=generation_type,
                generation_name=generation_name,
                generation_model_device=generator_device,
                generation_model_preload=generator_preload,
                generation_remote_url=generator_remote_url,
                generation_remote_token=generator_remote_token,
                generation_xinference_config=XinferenceConfig(
                    xinference_model_engine=xinference_model_engine,
                    xinference_model_format=xinference_model_format,
                    xinference_model_size=xinference_model_size,
                    xinference_mdoel_quantization=xinference_model_quantization,
                    xinference_ngpu=xinference_ngpu,
                ),
            ),
        )
        json.dump(config, open(save_path, "w", encoding="utf-8"))
        return "Config saved successfully"

    async def upload_files(
        self, config_path: str, upload_files: List[str], collection_name: str
    ) -> str:
        if self.pipeline:
            await self.pipeline.unload()
        self.pipeline = Pipeline(config_path)
        for upload_file in upload_files:
            docs = []
            is_sqlite = False
            if upload_file.endswith(".pdf"):
                docs = await PDFLoader(upload_file).load_and_split(
                    splitter=CharacterSplitter()
                )
            elif upload_file.endswith(".txt"):
                docs = await TextLoader(upload_file).load_and_split(
                    splitter=CharacterSplitter()
                )
            else:
                docs = await SqliteLoader(upload_file).load()
                is_sqlite = True
            if is_sqlite:
                embeds = await self.pipeline.embedding.get_embedding(docs)
                embed = await MeanPoolingCalculator.mean_pooling(embeds)
                insert_doc = Document(
                    page_content="\n".join([doc.page_content for doc in docs]),
                    metadata=docs[0].metadata if len(docs) > 0 else {},
                )
                await self.pipeline.store.add_document(
                    insert_doc, embed, collection_name
                )
            else:
                await self.pipeline.add_docs(docs, collection_name)
        return "Files uploaded successfully"

    async def recall(
        self,
        config_path: str,
        query: str,
        collection_name: str,
        upload_file: str = None,
    ) -> str:
        if self.pipeline:
            await self.pipeline.unload()
        self.pipeline = Pipeline(config_path)
        if upload_file:
            docs = await SqliteLoader(upload_file).load()
            embeds = await self.pipeline.embedding.get_embedding(docs)
            query_embed = (
                await self.pipeline.embedding.get_embedding(
                    docs=[Document(page_content=query)]
                )
            )[0]
            embeds.append(query_embed)
            embed = await MeanPoolingCalculator.mean_pooling(embeds)
            recall_res = await self.pipeline.store.search_by_embedding(
                embed, collection_name
            )
        else:
            recall_res = await self.pipeline.store.search_by_embedding(
                (
                    await self.pipeline.embedding.get_embedding(
                        docs=[Document(page_content=query)]
                    )
                )[0],
                collection_name,
            )

        return "\n\n".join(
            [f"src:{doc.metadata['src']}\n{doc.page_content}" for doc in recall_res]
        )

    async def generate_response(self, info: str, query: str) -> str:
        if not self.pipeline:
            return "Please recall first!"
        prompt = (
            "请根据下面的信息：\n"
            + "\n".join(info.split("\n")[:5])
            + "\n回答问题:"
            + query
        )
        return await self.pipeline.response_generator.generate(prompt)

    def init_app(self) -> gr.Blocks:
        with gr.Blocks() as app:
            with gr.Tab("Configuration"):
                with gr.Row():
                    with gr.Column():
                        config_label = gr.Label(
                            label="Configuration",
                            value="Configuration",
                        )
                        config_file = gr.File(
                            label="Upload config file",
                            file_types=["json"],
                        )

                with gr.Row():
                    with gr.Column():
                        embedding_model_label = gr.Label(
                            label="Embedding model configuration",
                            value="Embedding model configuration",
                        )
                        with gr.Row():
                            embedding_model = gr.Dropdown(
                                label="Select embedding model",
                                choices=self.__embedding_models,
                            )
                            embedding_model_name_or_path = gr.Textbox(
                                label="Model name or path",
                                placeholder="e.g. 'BAAI/bge-m3'",
                            )
                            embedding_model_device = gr.Radio(
                                label="Device",
                                choices=["cpu", "cuda"],
                            )
                            embedding_model_preload = gr.Checkbox(
                                label="Preload model",
                            )

                        with gr.Row():
                            embedding_remote_url = gr.Textbox(
                                label="Embedding Remote URL",
                                placeholder="e.g. 'http://localhost:9997'",
                            )
                            embedding_remote_token = gr.Textbox(
                                label="Embedding Remote token",
                                placeholder="your token or api-key here",
                            )

                    with gr.Column():
                        store_label = gr.Label(
                            label="Store configuration", value="Store configuration"
                        )
                        with gr.Row():
                            store_model = gr.Dropdown(
                                label="Select store model", choices=self.__store_models
                            )
                            store_local_path = gr.Textbox(
                                label="Local path",
                                placeholder="e.g. '/path/to/store'",
                            )

                        with gr.Row():
                            store_remote_url = gr.Textbox(
                                label="Store Remote URL",
                                placeholder="e.g. 'http://localhost:9997'",
                            )
                            store_remote_token = gr.Textbox(
                                label="Store Remote token",
                                placeholder="your token or api-key here",
                            )

                with gr.Row():
                    with gr.Row():
                        generator_label = gr.Label(
                            label="Response generator configuration",
                            value="Response generator configuration",
                        )

                with gr.Row():
                    with gr.Column():
                        generator = gr.Dropdown(
                            label="Select response generator",
                            choices=self.__response_generators,
                        )
                        generator_name_or_path = gr.Textbox(
                            label="Model name or path",
                        )
                        generator_device = gr.Radio(
                            label="Device",
                            choices=["cpu", "cuda"],
                        )
                        generator_preload = gr.Checkbox(label="Preload model")
                        generator_remote_url = gr.Textbox(
                            label="Response generator Remote URL",
                            placeholder="e.g. 'http://localhost:9997'",
                        )
                        generator_remote_token = gr.Textbox(
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
                        )[0]
                        xinference_model_quantization = (
                            gr.Textbox(
                                label="Model quantization",
                                placeholder="e.g. '4-bit', '8-bit'",
                            ),
                        )[0]
                        xinference_ngpu = (
                            gr.Textbox(
                                label="Number of GPUs",
                                placeholder="e.g. 'auto', '1', 'CPU'",
                            ),
                        )[0]

                save_path = gr.Textbox(label="Save path")
                save_button = gr.Button(value="Save")
                save_res = gr.Text(label="Save result", value="")

                config_file.change(
                    fn=self.set_config_from_file,
                    inputs=[config_file],
                    outputs=[
                        embedding_model,
                        embedding_model_name_or_path,
                        embedding_model_device,
                        embedding_model_preload,
                        embedding_remote_url,
                        embedding_remote_token,
                        store_model,
                        store_local_path,
                        store_remote_url,
                        store_remote_token,
                        generator,
                        generator_name_or_path,
                        generator_device,
                        generator_preload,
                        generator_remote_url,
                        generator_remote_token,
                        xinference_model_engine,
                        xinference_model_format,
                        xinference_model_size,
                        xinference_model_quantization,
                        xinference_ngpu,
                    ],
                )
                save_button.click(
                    fn=self.save_config_from_ui,
                    inputs=[
                        embedding_model,
                        embedding_model_name_or_path,
                        embedding_model_device,
                        embedding_model_preload,
                        embedding_remote_url,
                        embedding_remote_token,
                        store_model,
                        store_local_path,
                        store_remote_url,
                        store_remote_token,
                        generator,
                        generator_name_or_path,
                        generator_device,
                        generator_preload,
                        generator_remote_url,
                        generator_remote_token,
                        xinference_model_engine,
                        xinference_model_format,
                        xinference_model_size,
                        xinference_model_quantization,
                        xinference_ngpu,
                        save_path,
                    ],
                    outputs=[save_res],
                )

            with gr.Tab("Upload"):
                upload_config_file = gr.File(
                    label="Upload config file", file_types=["json"]
                )
                upload_files = gr.File(label="Upload files", file_count="multiple")
                upload_collection_name = gr.Textbox(
                    label="Collection name",
                    placeholder="e.g. 'default'",
                )
                upload_button = gr.Button(value="Upload")
                upload_res = gr.Text(label="Upload result", value="")

                upload_button.click(
                    fn=self.upload_files,
                    inputs=[upload_config_file, upload_files, upload_collection_name],
                    outputs=[upload_res],
                )

            with gr.Tab("Response"):
                response_config_file = gr.File(
                    label="Upload config file",
                    file_types=["json"],
                )
                response_file = gr.File(
                    label="Upload file",
                )
                response_query = gr.Textbox(
                    label="Query",
                    placeholder="e.g. 'What is the capital of France?'",
                )
                response_collection_name = gr.Textbox(
                    label="Collection name",
                    placeholder="e.g. 'default'",
                )
                response_recall_button = gr.Button(value="Recall")
                recall_res = gr.Text(label="Recall result", value="")
                response_button = gr.Button(value="Response")
                response_res = gr.Text(label="Response result", value="")

                response_recall_button.click(
                    fn=self.recall,
                    inputs=[
                        response_config_file,
                        response_query,
                        response_collection_name,
                        response_file,
                    ],
                    outputs=[recall_res],
                )
                response_button.click(
                    fn=self.generate_response,
                    inputs=[recall_res, response_query],
                    outputs=[response_res],
                )

            return app

    def launch(self):
        self.app.launch()
