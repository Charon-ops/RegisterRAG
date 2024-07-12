import requests
import re

from .base import WebExtractor
from entity.web_search_res import WebSearchRes
from logger import RagLogger


class FirecrawlExtractor(WebExtractor):
    def __init__(self, router_path: str, port: str):
        super().__init__()
        self.router_path = router_path
        self.port = port

    def extract(self, data: WebSearchRes):
        assert (
            data.source != "arxiv"
        ), f"source: {data.source} is not supported for firecrawl extractor."
        urls = data.urls
        extract_res = []
        for url in urls:
            try:
                response = requests.post(
                    url=f"{self.router_path}:{self.port}/v0/scrape",
                    headers={"Content-Type": "application/json"},
                    json={"url": url},
                )
                if response.status_code == 200 and response.json()["success"]:
                    content = response.json()["data"]["content"]
                    content = re.sub(r"\(data:.*?\)", "", content)
                    extract_res.append(content)
                else:
                    RagLogger().get_logger().error(
                        f"Error: {response.status_code}:{response.json()}"
                    )
                    extract_res.append("")
            except Exception as e:
                RagLogger().get_logger().error(f"Error: {e}")
                extract_res.append("")
                continue
        return extract_res
