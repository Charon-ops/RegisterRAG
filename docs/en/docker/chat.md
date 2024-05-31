# RegisterRag Zip Chat Service

## Start Service

To start the service, navigate to the docker directory and use the following command:

``` shell
sudo docker-compose up chat -d
```

If you need to force a rebuild, add the `--build` option:

``` shell
sudo docker-compose up chat -d --build
```

If you want to run in the foreground, remove the `-d` parameter

``` shell
sudo docker-compose up chat
```

## Chat API Overview

The Docker configuration runs `ollama`. It includes the `shenzhi-wang/Llama3-8B-Chinese-Chat-GGUF-8bit` model. For additional details and documentation, please visit the [official ollama repository](https://github.com/ollama/ollama).

Below is a Python example demonstrating how to interact with the API:

``` python
import requests
res = requests.post(
            url=f"{self.router_path}:{self.port}/api/generate",
            json={
                "model": "llama3-8b-chinese",
                "prompt": prompt,
                "stream": False,
            },
        )
print(res.json()["response"])
```

This example illustrates how to send a POST request to the API, using the provided model to generate a response based on the input prompt.
