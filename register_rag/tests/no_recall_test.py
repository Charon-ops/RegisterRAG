import sys

sys.path.append(f"{sys.path[0]}/..")

import os
import json
import re

from langchain_core.documents import Document
from scipy.spatial.distance import cosine

from response_gen import Llama3ResponseGen
from embedding import BgeEmbedding

responseGen = Llama3ResponseGen(router_path="http://localhost", port="10004")

with open(os.path.join(os.path.dirname(__file__), "qas.json"), "r") as f:
    qas = json.load(f)

query = []
response = []

for qa in qas:
    query.append(qa["query"])
    ans = qa["response"]
    match = re.search(r"问题回答：(.*?)(?:\n前置知识：|$)", ans, re.S)
    if match:
        response.append(match.group(1))
    else:
        print(f"Error: {qa['response']}")
        response.append("")

embedding = BgeEmbedding(router_path="http://localhost", port="10000")
query_embedding = embedding.embed_documents([Document(page_content=q) for q in query])

answer_embedding = embedding.embed_documents(
    [Document(page_content=r) for r in response]
)

generate_response = []

for i in range(len(query)):
    print(f"Process {i+1}/{len(query)}")
    prompt = query[i]
    res = responseGen.response_gen(prompt)
    if res == "":
        print(f"Error: {prompt}")
        continue
    print(f"Prompt: {prompt}")
    print(f"Response: {res}")
    generate_response.append(res)

generate_embedding = embedding.embed_documents(
    [Document(page_content=r) for r in generate_response]
)

ans_sim = 0.0
qa_sim = 0.0

for i in range(len(query)):
    dis = cosine(generate_embedding[i], answer_embedding[i])
    dis2 = cosine(query_embedding[i], generate_embedding[i])
    ans_sim += (1 - dis) / len(query)
    qa_sim += (1 - dis2) / len(query)

print(f"Average similarity: {ans_sim}")
print(f"Average similarity: {qa_sim}")

with open(os.path.join(os.path.dirname(__file__), "no_recall_eval.json"), "w") as f:
    json.dump({"ans_sim": ans_sim, "qa_sim": qa_sim}, f, ensure_ascii=False, indent=4)
