import sys

sys.path.append(f"{sys.path[0]}/..")

from web_extract.firecrawl_extractor import FirecrawlExtractor
from entity.web_search_res import WebSearchRes

test_data = WebSearchRes(
    query="123",
    source="baidu",
    urls=["https://baike.baidu.com/item/%E5%BE%AE%E7%94%9F%E7%89%A9/147527"],
)

firecrawl_extractor = FirecrawlExtractor(router_path="http://localhost", port="3002")

extract_res = firecrawl_extractor.extract(test_data)

print(extract_res)

print(type(extract_res[0]))
