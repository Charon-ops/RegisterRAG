from typing import List, Dict
import os
import json

from scipy.spatial.distance import cosine
from langchain_core.documents import Document

from .base import Evaluator
from ..app_register import AppRegister
from ..logger import RagLogger
from entity.evaluate_params import OurEvaluateConfig


class OurEvaluator(Evaluator):
    def __init__(self, config_path: str) -> None:
        super().__init__(config_path)
        self.config = OurEvaluateConfig(**self.config)
        self.recall_ans = self.load_recall_ans()
        return

    def load_recall_ans(self) -> List[Dict[str, List[str] | str]]:
        with open(self.config.recall_json_path, "r") as f:
            recall_ans = json.load(f)
        return recall_ans

    def evaluate(self, save_path: str) -> None:
        # Make sure the directory exists
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Create app through AppRegister
        app = AppRegister(self.config.app_name)

        query_embds = app.get_embeddings([Document(page_content=q) for q in self.query])

        recall_res = app.recall(
            query=self.query,
            query_embd=query_embds,
            retrieve_top_k=self.config.retrieve_top_k,
            rerank_top_k=self.config.rerank_top_k,
            store_name=self.config.store_name,
        )

        eval_res = {}
        eval_res["hitrate"] = (
            self.hitrate(recall_res) * 100
        )  # 召回结果在所有相关文档中的命中率，只要其中包含一个相关文档，或者是相关文档的一部分，就算命中
        eval_res["answer_accuracy"] = self.answer_accuracy(app, recall_res) * 100
        eval_res["content_accuracy"] = self.content_accuracy(recall_res) * 100
        eval_res["zipped_answer_accuracy"] = (
            self.zipped_answer_accuracy(app, recall_res) * 100
        )

    def hitrate(self, recall_res: List[List[Document]]) -> float:
        assert len(recall_res) == len(self.recall_ans)
        total_len = len(recall_res)
        hit = 0
        for i, each_query_recall in enumerate(recall_res):
            if i % 500 == 0:
                RagLogger().get_logger().info(
                    f"Calculating hitrate: {i}/{total_len}..."
                )
            current_hit = 0
            for i, doc in enumerate(each_query_recall):
                for ans in self.recall_ans[i]["recall_ans"]:
                    if doc.page_content in ans or ans in doc.page_content:
                        current_hit += 1
                current_hit /= len(self.recall_ans[i]["recall_ans"])
            hit += current_hit
        return hit / total_len

    def answer_accuracy(
        self, app: AppRegister, recall_res: List[List[Document]]
    ) -> float:
        assert len(recall_res) == len(self.recall_ans)
        total_len = len(recall_res)
        acc = 0.0
        for i, each_query_recall in enumerate(recall_res):
            if i % 50 == 0:
                RagLogger().get_logger().info(
                    f"Calculating answer accuracy: {i}/{total_len}..."
                )
            zipped_prompt = app.get_prompt(self.query[i], each_query_recall)
            response = app.get_response(zipped_prompt)
            response_embedding = app.get_embedding(response)
            ans_embedding = app.get_embedding(self.response[i])
            acc += 1 - cosine(response_embedding, ans_embedding)
        return acc / total_len

    def content_accuracy(self, recall_res: List[List[Document]]) -> float:
        total_len = len(recall_res)
        acc = 0.0

        for i, each_query_recall in enumerate(recall_res):
            if i % 500 == 0:
                RagLogger().get_logger().info(
                    f"Calculating content accuracy: {i}/{total_len}..."
                )
            current_acc = 0
            for i, doc in enumerate(each_query_recall):
                is_hit = False
                for ans in self.recall_ans[i]["recall_ans"]:
                    if doc.page_content in ans or ans in doc.page_content:
                        is_hit = True
                        break
                if is_hit:
                    current_acc += 1 / len(each_query_recall)
                else:
                    break
            acc += current_acc
        return acc / total_len

    def zipped_answer_accuracy(
        self, app: AppRegister, recall_res: List[List[Document]]
    ):
        total_len = len(recall_res)
        acc = 0.0

        for i, each_query_recall in enumerate(recall_res):
            if i % 50 == 0:
                RagLogger().get_logger().info(
                    f"Calculating zipped answer accuracy: {i}/{total_len}..."
                )
            zipped_prompt = f"请你根据下面的信息:{app.zip_prompt(self.query[i], each_query_recall)}\n回答问题:{self.query[i]}"
            response = app.get_response(zipped_prompt)
            response_embedding = app.get_embedding(response)
            ans_embedding = app.get_embedding(self.response[i])
            acc += 1 - cosine(response_embedding, ans_embedding)

        return acc / total_len
