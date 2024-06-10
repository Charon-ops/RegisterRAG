from typing import List, Dict

import random
import hashlib
import urllib
from http import client
import json
from concurrent.futures import ProcessPoolExecutor

import arxiv
from langchain_core.documents import Document
import spacy
from datasketch import MinHash, MinHashLSH

from .base import WebSearcher
from entity.web_search_res import WebSearchRes
from web_extract.base import WebExtractor
from web_extract import PdfExtractor


class ArxivSearcher(WebSearcher):
    def __init__(
        self, baidu_app_id: str | int, secret_key: str, max_threads: int = 10
    ) -> None:
        super().__init__()
        self.baidu_app_id = baidu_app_id
        self.secret_key = secret_key
        self.max_threads = max_threads

    def translate(self, query: str) -> str | None:
        sault = random.randint(32768, 65536)

        sign = f"{self.baidu_app_id}{query}{sault}{self.secret_key}"
        sign_md5 = hashlib.md5(sign.encode()).hexdigest()

        url = f"/api/trans/vip/translate?appid={self.baidu_app_id}&q={urllib.parse.quote(query)}&from=auto&to=en&salt={sault}&sign={sign_md5}"

        try:
            httpClient = client.HTTPConnection("fanyi-api.baidu.com")
            httpClient.request("GET", url)
            response = httpClient.getresponse()
            if response.status == 200:
                result = response.read().decode("utf-8")
                res_json = json.loads(result)
                return res_json["trans_result"][0]["dst"]
        except:
            return None

    def extract_keywords(self, text: str) -> List[str]:
        nlp = spacy.load("en_core_web_sm", exclude=["ner"])
        doc = nlp(text)
        keywords = [
            token.text
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and token.pos_ in {"NOUN", "VERB"}
        ]
        keywords_clean = [keyword for keyword in keywords if keyword.isalnum()]
        return keywords_clean

    def create_minhash(self, text: str) -> MinHash:
        keywords = self.extract_keywords(text)
        min_hash = MinHash(num_perm=64)
        for keyword in keywords:
            min_hash.update(keyword.encode("utf-8"))
        return min_hash

    def extract_revelant_docs(
        self, query: str, docs: List[Document], threshold: float = 0.07
    ) -> List[Document]:
        query_minhash = self.create_minhash(query)
        lsh = MinHashLSH(threshold=threshold, num_perm=64)

        lsh.insert("query", query_minhash)
        doc_minhashes = {}

        # for i, doc in enumerate(docs):
        #     doc_minhash = self.create_minhash(doc.page_content)
        #     doc_id = f"doc_{i}"
        #     lsh.insert(doc_id, doc_minhash)
        #     doc_minhashes[doc_id] = doc

        with ProcessPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [
                executor.submit(self.create_minhash, doc.page_content) for doc in docs
            ]
            for i, future in enumerate(futures):
                doc_id = f"doc_{i}"
                doc_minhash = future.result()
                lsh.insert(doc_id, doc_minhash)
                doc_minhashes[doc_id] = docs[i]

        similar_docs = lsh.query(query_minhash)
        revelant_docs = [
            doc_minhashes[doc_id] for doc_id in similar_docs if doc_id != "query"
        ]

        return revelant_docs

    def search(
        self,
        query: str,
        max_res: int = 3,
        start_page: int = -1,
        end_page: int = -1,
        extractor: WebExtractor = PdfExtractor(),
    ) -> List[Document]:
        query_english = self.translate(query)
        keywords = self.extract_keywords(query_english)
        word_num = max(3, len(keywords))
        query = "+AND+".join(keywords[:word_num])
        search_res = WebSearchRes(query=query, source="arxiv", urls=[])
        sorter = arxiv.SortCriterion.Relevance
        searcher = arxiv.Search(query, max_results=max_res, sort_by=sorter)
        for res in arxiv.Client().results(searcher):
            search_res.urls.append(res.pdf_url)
        extract_res = extractor.extract(search_res)
        revelant_docs = self.extract_revelant_docs(query_english, extract_res)
        return revelant_docs
