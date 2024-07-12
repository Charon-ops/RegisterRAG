# RegisterRag Store Docker Service

## Start Service

To start the service, navigate to the docker directory and use the following command:

``` shell
sudo docker-compose up store -d
```

If you need to force a rebuild, add the `--build` option:

``` shell
sudo docker-compose up store -d --build
```

If you want to run in the foreground, remove the `-d` parameter

``` shell
sudo docker-compose up store
```

## Store API

1. add documents:

   - Port: 10001

   - Path: /ann/add_docs

   - Request Body:

     ``` json
     {
       "doc_list": ["doc1", "doc2",...],
       "doc_emb_list": [
         [0.98112, 0.456465, ...],
         [...],
         [...],
         ...
       ],
       "doc_index": [0, 1, ...],
       "doc_name": "name",
       "doc_id": "id"
     }
     ```

     - `doc_list`: A list of strings to be stored.

     - `doc_emb_list`: A list of lists. Each list is the embedding of the corresponding text in the `doc_list` list.

     - `doc_index`: A list of integers. Each integer is the index of the corresponding text in the `doc_list` list.

     - `doc_name`: A string specifying the name of the document.

     - `doc_id`: A string specifying the id of the document.

      **Note:** The lengths of `doc_list`, `doc_emb_list`, and `doc_index` must be the same. The `doc_name` is like the title of a book, and the `doc_list` is like the content of the book's chapters. The `doc_id` is like the ISBN of the book. When adding a document, you should provide `doc_list`, `doc_emb_list`, `doc_index`, and either `doc_name` or `doc_id`.

     If you don't provide a `doc_id`, the system will generate one based on the `doc_name`. If the `doc_name` already exists, the system will update the document with that name.

     When using the store service, `doc_id` is not required, but `doc_name` is required. If two different documents have the same `doc_name`, the system will update the document with that name. This means that `doc_name` **should be unique**.

   - Response: None

   - Example Usage:

     ``` python
      sentences = ["apple", "banana"]
     
      embs = requests.post(
          url="http://localhost:10000/get_embedding",
          json={"text": sentences, "model_type": "bge"},
      ).json()["embedding"]
     
      requests.post(
          url="http://localhost:10001/ann/add_docs",
          json={
              "doc_list": sentences,
              "doc_emb_list": embs,
              "doc_index": [0, 1],
              "doc_name": "test",
          },
      )
     ```

2. search_by_embedding:

   - Port: 10001

   - Path: /ann/search_by_embedding

   - Request Body:

     ``` json
     {
       "emb": [0.98112, 0.456465, ...],
       "num": 10
     }
     ```

     - `emb`: A list of floats. The embedding of the query text.

     - `num`: An integer specifying the number of documents to return.

   - Response:

     ```json
     {
       "knowledges": [
         "str1",
         "str2",
         ...
       ]
     }
     ```

     - `knowledges`: A list of strings. The documents that are most similar to the query text.

     **Note:** When retrieving the first chunk, the system will also try to include its neighboring chunks. If the passage along with its neighbors has a closer distance to the query, the neighbors will be added to the result.

     - Example Usage:

       ``` python
        search_sentences = ["app"]
        search_embs = requests.post(
            url="http://localhost:10000/get_embedding/",
            json={"text": search_sentences, "model_type": "bge"},
        ).json()["embedding"]
       
        search_res = requests.post(
            url="http://localhost:10001/ann/search",
            json={"query_vec": search_embs[0], "num": 2},
        ).json()["knowledges"]
       ```

3. get_ids:

   - Port: 10001

   - Path: /ann/get_ids

   - Request Body:

     ```json
     {
       "docs": [
         "apple",
         "banana",
         "..."
       ]
     }
     ```

     - `docs`: A list of strings. The contents to get the ids.

   - Response:

       ```json
       {
         "ids": [
           0,
           1,
           ...
         ]
       }
       ```

       - `ids`: A list of integers. The ids of the chunks.

4. remove_items:

   - Port: 10001

   - path: /ann/remove_items

   - Request Body

     ```json
     {
         "ids": [0, 1, ...]
     }
     ```

     - `ids`: A list of integers. The ids of the chunks to be removed.
