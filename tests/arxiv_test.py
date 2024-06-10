import sys
import time

sys.path.append(f"{sys.path[0]}/..")


from websearch import ArxivCallback

baidu_app_id = 123  # 请更换为自己的百度翻译API的app_id
baidu_api_key = "your_secret_key"  # 请更换为自己的百度翻译API的secret_key

query = "什么是微生物？"

arxiv_searcher = ArxivCallback(baidu_app_id, baidu_api_key)

start_time = time.time()
arxiv_res = arxiv_searcher.search(query, max_res=3)
end_time = time.time()

print(arxiv_res)
print(f"Time: {end_time - start_time}")
