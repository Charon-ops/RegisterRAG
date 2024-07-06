import os
import requests

file_path = os.path.join(os.path.dirname(__file__), "merge.txt")

with open(file_path, "r") as f:
    insert_content = f.read()

response = requests.post(
    url="http://localhost:20000/insertDocToStore",
    json={
        "file_content": insert_content,
        "file_type": "text",
        "app_name": "wisdomentor",
        "begin_index": 0,
        "doc_name": "microbiology",
        "store_name": "wisdomentor",
    },
)

if response.status_code == 200:
    print("Insert successfully!")
