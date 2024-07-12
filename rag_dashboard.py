from typing import List

import gradio as gr


selected_logs = []
logs = []


def check_valid_url(url: str) -> bool:
    return url.strip().startswith("http")


def split_file(file_path: str) -> List[str]:
    global selected_logs
    global logs
    selected_logs = []
    logs = []
    with open(file_path, "r") as f:
        logs = f.readlines()
    return logs


def handle_select(index: List[int]):
    print(f"index: {index}")


def upload_docs(
    embedding_model: str,
    embedding_remote_url: str,
    store: str,
    store_remote_url: str,
    upload_files: List[str],
):
    if not check_valid_url(embedding_remote_url) or not check_valid_url(
        store_remote_url
    ):
        return "Invalid remote url"
    print(f"upload_files: {upload_files}")
    return "Process success!"


def recall_docs(
    embedding_model: str,
    embedding_remote_url: str,
    store: str,
    store_remote_url: str,
    upload_file: str,
) -> List[str]:
    if not check_valid_url(embedding_remote_url) or not check_valid_url(
        store_remote_url
    ):
        return "Invalid remote url"
    return ""


def generate_reponse(
    embedding_model: str,
    embedding_remote_url: str,
    store: str,
    store_remote_url: str,
    qwen_api_key: str,
    upload_file: str,
    query: str,
) -> str:
    if not check_valid_url(embedding_remote_url) or not check_valid_url(
        store_remote_url
    ):
        return "Invalid remote url"
    return ""


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
            upload_store_remote_url = gr.Textbox(label="Store remote url")

            upload_button = gr.Button("Upload")

        upload_files = gr.Files(label="Q&A Files")
        upload_output = gr.Textbox(label="Upload Output")
        upload_button.click(
            fn=upload_docs,
            inputs=[
                upload_embedding_dropdown,
                upload_embedding_remote_url,
                upload_store_dropdown,
                upload_store_remote_url,
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
            response_store_remote_url = gr.Textbox(label="Store remote url")

            recall_button = gr.Button("Recall")

        with gr.Row():
            response_query = gr.Textbox(label="Query")
            response_upload_file = gr.File(label="Upload File")

        with gr.Row():
            recall_res = gr.Textbox(label="Recall Result")

        with gr.Row():
            response_button = gr.Button("Response")
            response_res = gr.Textbox(label="Response Result")

        recall_button.click()

app.launch()
