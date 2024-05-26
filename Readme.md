# Register RAG

<h4 align=center>
<p>Quickly configure the RAG framework to meet your needs via JSON</p>
</h4>

## Category
- [📣 Updates](#updates)
- [📣 Deploy](#deploy)
- [📣 Usage](#usage)

## 📣 Updates
- 2024-05-26 We released the first version of the base RAG framework that can support configurations

## 📣 Deploy

Clone code from our repository

```bash
git clone https://github.com/Charon-ops/RegisterRAG.git
```

Start the embedding, store, rerank, prompt zip and generation service via docker-compose.yml. If you don't have a previous image for these services, the required image will be automatically built here.

```bash
cd docker
docker compose up
cd ..
```

Wait for the container to start successfully, register rag deployed successfully

## 📣 Usage

Install the required packages

```bash
pip install -r requirments.txt
```

If you want to customise your own RAG framework, you can configure it by modifying the app_register_config.json file. See the [Config](#config) section for details on configuring the fields in the json file.
```bash
vim app_register_config.json
```

Start the Register RAG service (default port is 8000)

```bash
uvicorn service:service
```

## 📣 Config

Specific configuration details are detailed in the following demo code block. You can modify the name field and configure the corresponding args field correctly. Please refer to the config_reference.md file for the detailed field correspondences.

```python
{
    "app_name": {
        # config your repository database
        "database": [
            {
                "name": "ann",
                "args": {
                    "router_path": "your service url" ,
                    "port": "your service port"
                }
            }
        ],
        # config knowledge loader method
        "loader": {
            "name": "text",
            "args": {}
        },
        # config the embedding method
        "embedding": {
            "name": "bge",
            "args": {
                "router_path": "your service url",
                "port": "your service port"
            }
        },
        # config the splitter method
        "splitter": {
            "name": "recursive",
            "args": {
                "chunk_size": 400
            }
        },
        # config the web search method, this item can be an empty list
        "websearch": [
            {
                "name": "arxiv",
                "args": {}
            },
            {
                "name": "baidu",
                "args": {}
            }
        ],
        # config the rerank method
        "rerank": {
            "name": "bge_normal",
            "args": {
                "router_path": "your service url",
                "port": "your service port"
                "path": "bge_normal_rerank"
            }
        },
        # config the prompt generation template, args must be an empty dict
        "prompt_gen": {
            "name": "base",
            "args": {}
        },
        # config the prompt zip method, it wil be tricked when the prompt is too long
        "prompt_zip": {
            "name": "llmlingua2",
            "args": {
                "router_path": "your service url",
                "port": "your service port"
                "path": "llmlingua2"
            }
        },
        # config the chat model, it only supports the local model currently
        "response_gen": {
            "name": "llama3",
            "args": {
                "router_path": "your service url",
                "port": "your service port"
            }
        }
    }
}
```