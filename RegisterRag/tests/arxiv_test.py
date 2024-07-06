import sys
import time

sys.path.append(f"{sys.path[0]}/..")


from websearch import ArxivCallback

baidu_app_id = "your_app_id"  # 请更换为自己的百度翻译API的app_id
baidu_api_key = "your_secret_key"  # 请更换为自己的百度翻译API的secret_key
qwen_plus_api_key = "you_api_key"  # 请更换为自己的Qwen-Plus API的api_key

query = "什么是微生物？"

arxiv_searcher = ArxivCallback(qwen_plus_api_key, baidu_app_id, baidu_api_key)

start_time = time.time()
arxiv_res = arxiv_searcher.search(query, max_res=3)
end_time = time.time()

print("\n\n".join([res.page_content for res in arxiv_res]))
print(f"Time: {end_time - start_time}")
