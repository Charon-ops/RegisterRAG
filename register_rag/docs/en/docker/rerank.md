# RegisterRag Rerank Docker Service

## Start Service

To start the service, navigate to the docker directory and use the following command:

``` shell
sudo docker-compose up rerank -d
```

If you need to force a rebuild, add the `--build` option:

``` shell
sudo docker-compose up rerank -d --build
```

If you want to run in the foreground, remove the `-d` parameter

``` shell
sudo docker-compose up rerank
```

## Rerank API

1. Bge Normal Rerank:

   - Port: 10002

   - Path: /bge_normal_rerank

   - Request Body:

     ``` json
     {
         "query": "What is an apple?",
         "documents": [
             "An apple is ...",
             "Apples are ...",
             ...
         ]
     }
     ```

     - `query`: A string representing either the user's original question or a system-rewritten version.
     - `documents`: A list of strings, comprising documents retrieved by the store service.

   - Response:

     ```json
     {
         "reranked_documents": [
             "Apples are ...",
             "An apple is ...",
             ...
         ]
     }
     ```

     - `reranked_documents`: A list of strings, ordered from most to least relevant.

   - Example Usage:

     ``` python
     import requests
     
     query = "什么是微生物？"
     
     docs = [
         "计算机是一种用于计算的电子设备。",
         "微生物（microorganism）是生活在环境中的微小生物，它们可以对环境产生影响，并参与生物圈的形成。",
         "微生物是生命体中非常微小的个体，它们可以存在于任何地方，从土壤到空气，从海洋到高山。",
         "微生物可以参与各种生物过程，如呼吸作用、光合作用和厌氧呼吸作用。",
         "微生物可以对环境产生影响，如降解污染物、促进土壤肥力、控制病毒传播等。",
         "微生物是生命体中非常重要的组成部分，它们共同构成了生态系统。",
         "计算机科学（Computer Science）是研究计算机系统结构和行为的学科。",
         "计算机科学是人工智能、网络安全、大数据分析等领域的基石。",
     ]
     
     res = requests.post(
         url="http://localhost:10002/bge_normal_rerank",
         json={
             "query": query,
             "documents": docs,
         },
     ).json()["reranked_documents"]
     
     print(res)
     ```
