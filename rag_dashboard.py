from typing import List
import os

import gradio as gr
from xinference_client import RESTfulClient
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from response_gen import QwenPlusResponseGen
from store import ChromaStore
from loader import SqliteLoader


selected_logs = []
logs = []


def check_valid_url(url: str) -> bool:
    return url.strip().startswith("http")


def convert_path(path: str) -> str:
    if not path.startswith("/"):
        return os.path.join(os.path.dirname(__file__), path)
    else:
        return path


def get_embedding(text: str, remote_url: str) -> List[float]:
    client = RESTfulClient(remote_url)
    model = client.get_model("bge-m3")
    embed_res = model.create_embedding([text])
    return embed_res["data"][0]["embedding"]


def handle_select(index: List[int]):
    print(f"index: {index}")


def upload_docs(
    embedding_model: str,
    embedding_remote_url: str,
    store: str,
    store_path: str,
    collection_name: str,
    upload_files: List[str],
):
    if not check_valid_url(embedding_remote_url):
        return "Invalid remote url"
    if embedding_model != "bge":
        return ValueError("Only support bge model")
    if store != "Chroma":
        return ValueError("Only support Chroma store")
    store_path = convert_path(store_path)
    from_sql = False
    for file in upload_files:
        if file.endswith("pdf"):
            loader = PyPDFLoader(file)
            contents = loader.load_and_split(
                text_splitter=RecursiveCharacterTextSplitter()
            )
        elif file.endswith("txt"):
            loader = TextLoader(file)
            contents = loader.load_and_split(
                text_splitter=RecursiveCharacterTextSplitter()
            )
        else:
            from_sql = True
            loader = SqliteLoader(file)
            contents = loader.load_file()
        chromaStore = ChromaStore(os.path.join(os.path.dirname(__file__), store_path))
        chromaStore.add_documents(
            documents=contents,
            embedding_remote_url=embedding_remote_url,
            collection_name=collection_name,
            from_sql=from_sql,
        )
    return "Process success!"


def recall_docs(
    embedding_model: str,
    embedding_remote_url: str,
    store: str,
    store_path: str,
    query: str,
    collection_name: str,
    upload_file: str,
    top_k: int = 5,
) -> str:
    if not check_valid_url(embedding_remote_url):
        return "Invalid remote url"
    if embedding_model != "bge":
        raise ValueError("Only support bge model")
    if store != "Chroma":
        raise ValueError("Only support Chroma store")
    if not os.path.exists(upload_file):
        raise ValueError("Upload file not exists")
    store_path = convert_path(store_path)
    if upload_file.endswith(".txt"):
        loader = TextLoader(upload_file)
        logs = loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter())
    else:
        loader = SqliteLoader(upload_file)
        logs = loader.load_file()
    log_embeds = [get_embedding(log.page_content, embedding_remote_url) for log in logs]
    log_embed = [sum(x) / len(x) for x in zip(*log_embeds)]
    # query_embed = get_embedding(query, embedding_remote_url)
    # for i in range(len(query_embed)):
    #     query_embed[i] = (query_embed[i] + log_embed[i]) / 2
    chromaStore = ChromaStore(os.path.join(os.path.dirname(__file__), store_path))
    res = chromaStore.search_by_embed(
        log_embed, collction_name=collection_name, results=top_k
    )
    metadatas = res["metadatas"][0]
    res_str = ""
    for metadata in metadatas:
        res_str += metadata["source"] + "\n"
    return res_str


def generate_reponse(
    recall_res: str,
    qwen_api_key: str,
    query: str,
) -> str:
    qwen_plus_response_gen = QwenPlusResponseGen(qwen_api_key)
    template = "请根据以下信息回答问题：\n" + recall_res + "\n问题：" + query
    return qwen_plus_response_gen.response_gen(template)


with gr.Blocks() as app:
    with gr.Tab("Upload"):
        with gr.Row():
            upload_embedding_dropdown = gr.Dropdown(
                label="Embedding Model", choices=["bge", "bert"], value="bge"
            )
            upload_embedding_remote_url = gr.Textbox(label="Embedding remote url")

            upload_store_dropdown = gr.Dropdown(
                label="Store", choices=["Chroma"], value="Chroma"
            )
            upload_store_db_path = gr.Textbox(label="Store db path")

            upload_collection_name = gr.Textbox(label="Collection Name")

            upload_button = gr.Button("Upload")

        upload_files = gr.Files(label="Q&A Files")
        upload_output = gr.Textbox(label="Upload Output")
        upload_button.click(
            fn=upload_docs,
            inputs=[
                upload_embedding_dropdown,
                upload_embedding_remote_url,
                upload_store_dropdown,
                upload_store_db_path,
                upload_collection_name,
                upload_files,
            ],
            outputs=[upload_output],
        )

    with gr.Tab("Response"):
        with gr.Row():
            response_embedding_dropdown = gr.Dropdown(
                label="Embedding Model", choices=["bge", "bert"], value="bge"
            )
            response_embedding_remote_url = gr.Textbox(label="Embedding remote url")

            response_store_dropdown = gr.Dropdown(
                label="Store", choices=["Chroma"], value="Chroma"
            )
            response_store_db_path = gr.Textbox(label="Store db path")

            response_collection_name = gr.Textbox(label="Collection Name")

            recall_button = gr.Button("Recall")

        with gr.Row():
            response_upload_file = gr.File(label="Upload File")

        with gr.Row():
            recall_res = gr.Textbox(label="Recall Result")

        with gr.Row():
            response_query = gr.Textbox(label="Query")
            qwen_plus_api_key = gr.Textbox(label="Qwen Plus API Key")
            response_button = gr.Button("Response")

        response_res = gr.Textbox(label="Response Result")

        recall_button.click(
            fn=recall_docs,
            inputs=[
                response_embedding_dropdown,
                response_embedding_remote_url,
                response_store_dropdown,
                response_store_db_path,
                response_query,
                response_collection_name,
                response_upload_file,
            ],
            outputs=[recall_res],
        )

        response_button.click(
            fn=generate_reponse,
            inputs=[recall_res, qwen_plus_api_key, response_query],
            outputs=[response_res],
        )

if __name__ == "__main__":
    app.launch()
