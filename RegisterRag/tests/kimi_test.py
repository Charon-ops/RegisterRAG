import sys

sys.path.append(f"{sys.path[0]}/..")

import brotli

from web_extract import KimiExtractor
from entity.web_search_res import WebSearchRes

kimiExtractor = KimiExtractor()

data = WebSearchRes(
    query="请你根据这个问题：如何学习python，总结这个链接的内容。",
    source="baidu",
    urls=["https://www.runoob.com/python/python-tutorial.html"],
)

print(kimiExtractor.extract(data))
