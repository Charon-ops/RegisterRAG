# Register RAG

<h4 align=center>
<p>Quickly configure the RAG framework to meet your needs via JSON</p>
</h4>

## Category

- [Register RAG](#register-rag)
  - [Category](#category)
  - [🔔 Updates](#-updates)
  - [🚀 Deploy](#-deploy)
    - [Making the necessary modifications](#making-the-necessary-modifications)
    - [Starting the Service](#starting-the-service)
    - [Running in Detached Mode](#running-in-detached-mode)
    - [Building Process](#building-process)
    - [Starting Specific Services](#starting-specific-services)
    - [Stopping the Service](#stopping-the-service)
  - [📈 Usage](#-usage)
  - [⚙️ Config](#️-config)

## 🔔 Updates

- 2024-05-26 We released the first version of the base RAG framework that can support configurations

## 🚀 Deploy

### Making the necessary modifications

Clone code from our repository

```bash
git clone https://github.com/Charon-ops/RegisterRAG.git
```

**Before launching the service, please modify the `docker-compose.yml` file according to your GPU configuration.** Our server is equipped with four graphics cards, and we have deployed all services except for the `zip` service on the third card. You will need to adjust the `NVIDIA_VISIBLE_DEVICES` setting to match the GPU configuration of your own computer.

At the same time, you also need to **modify the mount point of the store service**. Specifically, change the `volumes` section in the `store` service to `path/to/your/data:/app/data`.

### Starting the Service

After making the necessary modifications, you can start the service using the `docker-compose.yml` file. If you haven't previously built the process, it will automatically build without any manual intervention required.

```bash
cd docker
docker-compose up
cd ..
```

Wait for the container to start successfully, register rag deployed successfully.

### Running in Detached Mode

If you prefer not to automatically attach to the container, you can add the `-d` parameter to run in detached mode. For example:

```bash
cd docker
docker-compose up -d
cd ..
```

### Building Process

The building process involves downloading weights, which by default uses the [hf-mirror](https://hf-mirror.com/) repository. You can modify the `HF_ENDPOINT` setting in the Dockerfile to better suit your network conditions. Please note, due to the substantial size of the weights, this process might take some time.

### Starting Specific Services

If you prefer to start only certain services, use the command below:

``` bash
docker-compose up embedding
```

This will launch only the `embedding` service.

### Stopping the Service

When you need to stop the service, navigate to the Docker directory and execute the following command:

```bash
cd docker
docker-compose down
```

This will terminate all running containers associated with the service.

## 📈 Usage

Install the required packages

```bash
pip install -r requirments.txt
```

If you want to customise your own RAG framework, you can configure it by modifying the app_register_config.json file. See the [Config](#️-config) section for details on configuring the fields in the json file.

```bash
vim app_register_config.json
```

Start the Register RAG service (default port is 8000)

```bash
uvicorn service:service
```

## ⚙️ Config

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
                "port": "your service port",
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
                "port": "your service port",
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
