import gradio as gr
import os

from langchain_core.documents import Document

from app_register import AppRegister


class RagDashboard:
    def __init__(self) -> None:
        pass

    def launch(self):
        with gr.Blocks() as rag_blocks:
            with gr.Tab("Data Processing"):
                config_file_input1 = gr.File(label="Config File Path", type="filepath")
                data_files_input = gr.Files(label="Data Path", type="filepath")
                app_name1 = gr.Text(label="App Name")
                process_button = gr.Button("Process Data")
                process_output = gr.Textbox(label="Output")
                process_button.click(
                    self.process_data,
                    inputs=[config_file_input1, data_files_input, app_name1],
                    outputs=process_output,
                )

            with gr.Tab("Response Generation"):
                config_file_input2 = gr.File(label="Config File Path", type="filepath")
                query_input = gr.Textbox(label="Query")
                app_name2 = gr.Text(label="App Name")
                generate_button = gr.Button("Generate Response")
                response_output = gr.Textbox(label="Response")
                generate_button.click(
                    self.get_response,
                    inputs=[config_file_input2, query_input, app_name2],
                    outputs=response_output,
                )

        rag_blocks.launch()

    def process_data(self, config_path, data_paths, app_name) -> None:
        app = AppRegister(app_name, config_path)
        data = []
        for file in os.walk(data_paths):
            if os.path.isdir(file):
                continue
            with open(file, "r") as f:
                data.append(f.read())
        app.add_database(
            [Document(page_content=d) for d in data], app.get_embeddings(data)
        )

    def get_response(self, config_path, query, app_name) -> str:
        app = AppRegister(app_name, config_path)
        response = app.get_response(query)
        return response


dashboard = RagDashboard()
dashboard.launch()
