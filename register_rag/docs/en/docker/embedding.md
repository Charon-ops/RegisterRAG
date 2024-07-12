# RegisterRag Embedding Docker Service

## Start Service

To start the service, navigate to the docker directory and use the following command:

``` shell
sudo docker-compose up embedding -d
```

If you need to force a rebuild, add the `--build` option:

``` shell
sudo docker-compose up embedding -d --build
```

If you want to run in the foreground, remove the `-d` parameter

``` shell
sudo docker-compose up embedding
```

## Embedding API

- Port: 10000

- Path: /get_embedding

- Request Body:

  ```json
  {
    "text": [
      "text1",
      "text2",
      ...
    ],
    "model_type": "bge" | "bert"
  }
  ```

  - `text`: A list of strings to be embedded.
  - `model_type`: A string specifying the embedding model. Currently supports `bge` and `bert`.

    **Note**: The dimension of the `bge` embedding is 1024, and the dimension of the `bert` embedding is 768. **The default dimension in our storage service is 1024**. If you want to use the bert embedding, you should change the dimension setting in the storage service.

- Response:

  ``` json
  {
    "embedding": [
      [0.98112, 0.456465, ...],
      [...],
      [...],
      ...
    ]
  }
  ```

  - `embedding` is a list of lists. Each list is the embedding of the corresponding text in the `text` list.

- Example Usage:

  To get embeddings for a list of strings:

  ``` python
  embs = requests.post(
            url="http://localhost:10000/get_embedding/",
            json={
                "text": ["hello", "world"],
                "model_type": "bge"
            }
        ).json()["embedding"]
  ```

  To get embeddings for a list of `Document` objects:

  ``` python
  from langchain_core.documents import Document 
  documents: List[Document] = [Document(page_content = "hello", metadata = {}), Document(page_content = "world", metadata = {})]
  embs = requests.post(
            url="http://localhost:10000/get_embedding/",
            json={
                "text": [doc.page_content for doc in documents],
                "model_type": "bge"
            }
        ).json()["embedding"]
  ```
