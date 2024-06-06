from typing import List, Dict, Tuple
import os
import json
import time

from scipy.spatial.distance import cosine
from langchain_core.documents import Document

from .base import Evaluator
from app_register import AppRegister
from logger import RagLogger
from .dataset_loader import OurDatasetLoader
from entity.evaluate_params import OurEvaluateConfig


class OurEvaluator(Evaluator):
    def __init__(self, config_path: str) -> None:
        super().__init__(config_path)
        self.config = OurEvaluateConfig(**self.config)
        self.query, self.ans, self.recall_ans = OurDatasetLoader(
            self.config.data_path
        ).load()
        self.generate_res = []
        self.zipped_generate_res = []

    def evaluate(self, save_path: str) -> None:
        tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        # Make sure the directory exists
        save_dir = "/".join(save_path.split("/")[:-2])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Create app through AppRegister
        app = AppRegister(self.config.app_name)

        query_embedding_file = os.path.join(tmp_dir, "query_embedding.json")
        if not os.path.exists(query_embedding_file):
            query_embedding_start_time = time.time()
            query_embds = app.get_embeddings(
                [Document(page_content=q) for q in self.query]
            )
            query_embedding_end_time = time.time()
            query_embedding_time = query_embedding_end_time - query_embedding_start_time
            with open(query_embedding_file, "w") as f:
                json.dump(
                    {"time": query_embedding_time, "embds": query_embds}, f, indent=4
                )
        else:
            with open(query_embedding_file, "r") as f:
                data = json.load(f)
            query_embedding_time = data["time"]
            query_embds = data["embds"]

        recall_res_file = os.path.join(tmp_dir, "recall_res.json")
        if not os.path.exists(recall_res_file):
            recall_start_time = time.time()
            recall_res = app.recall(
                query=self.query,
                query_embd=query_embds,
                retrieve_top_k=self.config.retrieve_top_k,
                rerank_top_k=self.config.rerank_top_k,
                store_name=self.config.store_name,
            )
            recall_end_time = time.time()
            recall_time = recall_end_time - recall_start_time
            recall_content = []
            for r in recall_res:
                recall_content.append([item.page_content for item in r])
            with open(recall_res_file, "w") as f:
                json.dump({"time": recall_time, "res": recall_content}, f, indent=4)
        else:
            with open(recall_res_file, "r") as f:
                data = json.load(f)
            recall_content = data["res"]
            recall_res = []
            for content in recall_content:
                recall_res.append([Document(page_content=item) for item in content])
            recall_time = data["time"]

        eval_res = {}
        eval_res["retrieve_precision"] = 0.0
        eval_res["retrieve_recall"] = 0.0
        eval_res["retrieve_f1"] = 0.0
        eval_res["retrieve_hitrate"] = 0.0
        eval_res["retrieve_precision"],
        eval_res["retrieve_recall"],
        eval_res["retrieve_f1"],
        eval_res["retrieve_hitrate"] = self.retrieve_eval(recall_res) * 100
        eval_res["content_precision"] = self.content_precision(recall_res) * 100

        generation_file_path = os.path.join(tmp_dir, "generation_res.json")
        if not os.path.exists(generation_file_path):
            generation_start_time = time.time()
            eval_res["answer_sim"] = self.answer_sim(app, recall_res) * 100
            generation_end_time = time.time()
            generation_time = generation_end_time - generation_start_time
            with open(generation_file_path, "w") as f:
                json.dump(
                    {
                        "time": generation_time,
                        "res": self.generate_res,
                        "sim": eval_res["answer_sim"],
                    },
                    f,
                    indent=4,
                )
        else:
            with open(generation_file_path, "r") as f:
                data = json.load(f)
            self.generate_res = data["res"]
            eval_res["answer_sim"] = data["sim"]
            generation_time = data["time"]

        zipped_generation_file_path = os.path.join(
            tmp_dir, "zipped_generation_res.json"
        )
        if not os.path.exists(zipped_generation_file_path):
            zipped_generation_start_time = time.time()
            eval_res["zipped_answer_sim"] = (
                self.zipped_answer_sim(app, recall_res) * 100
            )
            zipped_generation_end_time = time.time()
            zipped_generation_time = (
                zipped_generation_end_time - zipped_generation_start_time
            )
            with open(zipped_generation_file_path, "w") as f:
                json.dump(
                    {
                        "time": zipped_generation_time,
                        "res": self.zipped_generate_res,
                        "sim": eval_res["zipped_answer_sim"],
                    },
                    f,
                    indent=4,
                )
        else:
            with open(zipped_generation_file_path, "r") as f:
                data = json.load(f)
            self.zipped_generate_res = data["res"]
            eval_res["zipped_answer_sim"] = data["sim"]
            zipped_generation_time = data["time"]

        eval_res["answer_accuracy"] = self.answer_accuracy(app, recall_res) * 100
        eval_res["qa_revelance"] = self.qa_relevance(app, recall_res) * 100
        eval_res["query_embedding_time"] = query_embedding_time / len(self.query) * 100
        eval_res["recall_time"] = recall_time / len(self.query)
        eval_res["generation_time"] = generation_time / len(self.query)
        eval_res["zipped_generation_time"] = zipped_generation_time / len(self.query)

        with open(save_path, "w") as f:
            json.dump(eval_res, f)

    def retrieve_eval(self, recall_res: List[List[Document]]) -> float:
        total_len = len(recall_res)

        precision = 0.0
        recall = 0.0
        f1 = 0.0
        hitrate = 0.0

        for i, each_query_recall in enumerate(recall_res):
            if i % 100 == 0:
                RagLogger().get_logger().info(
                    f"Calculating retrieve accuracy: {i}/{total_len}..."
                )
                is_reaclled = [False] * len(each_query_recall)
                hit_num = 0
                for ans in self.recall_ans[i]:
                    for j, doc in enumerate(each_query_recall):
                        if doc.page_content in ans or ans in doc.page_content:
                            is_reaclled[j] = True
                            hit_num += 1
                            break
                current_tp = sum(is_reaclled)
                current_fp = len(each_query_recall) - current_tp
                current_fn = len(self.recall_ans[i]) - current_tp
                current_precision = current_tp / (current_tp + current_fp)
                current_recall = current_tp / (current_tp + current_fn)
                current_f1 = (
                    2
                    * current_precision
                    * current_recall
                    / (current_precision + current_recall)
                )
                precision += current_precision / total_len
                recall += current_recall / total_len
                f1 += current_f1 / total_len
                hitrate += hit_num / len(self.recall_ans[i]) / total_len
        return precision, recall, f1, hitrate

    def answer_sim(self, app: AppRegister, recall_res: List[List[Document]]) -> float:
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
            self.generate_res.append(response)
            response_embedding = app.get_embedding(response)
            ans_embedding = app.get_embedding(self.ans[i])
            acc += 1 - cosine(response_embedding, ans_embedding)
        return acc / total_len

    def content_precision(self, recall_res: List[List[Document]]) -> float:
        total_len = len(recall_res)
        acc = 0.0

        for i, each_query_recall in enumerate(recall_res):
            if i % 100 == 0:
                RagLogger().get_logger().info(
                    f"Calculating content accuracy: {i}/{total_len}..."
                )
            current_acc = 0
            for j, doc in enumerate(each_query_recall):
                is_hit = False
                for ans in self.recall_ans[i]:
                    if doc.page_content in ans or ans in doc.page_content:
                        is_hit = True
                        break
                if is_hit:
                    current_acc += 1 / len(each_query_recall)
                else:
                    break
            acc += current_acc
        return acc / total_len

    def zipped_answer_sim(self, app: AppRegister, recall_res: List[List[Document]]):
        assert len(recall_res) == len(self.recall_ans)
        total_len = len(recall_res)
        acc = 0.0

        for i, each_query_recall in enumerate(recall_res):
            if i % 50 == 0:
                RagLogger().get_logger().info(
                    f"Calculating zipped answer accuracy: {i}/{total_len}..."
                )
            zipped_prompt = f"请你根据下面的信息:{app.zip_prompt(self.query[i], each_query_recall)}\n回答问题:{self.query[i]}"
            response = app.get_response(zipped_prompt)
            self.zipped_generate_res.append(response)
            response_embedding = app.get_embedding(response)
            ans_embedding = app.get_embedding(self.ans[i])
            acc += 1 - cosine(response_embedding, ans_embedding)

        return acc / total_len

    def answer_accuracy(self, app: AppRegister, recall_res: List[List[Document]]):
        assert len(recall_res) == len(self.recall_ans)
        total_len = len(recall_res)
        right_num = 0

        for i in range(total_len):
            ans = self.ans[i]
            generate_res = self.generate_res[i]
            ans_embedding = app.get_embedding(ans)
            generate_res_embedding = app.get_embedding(generate_res)
            dis = cosine(ans_embedding, generate_res_embedding)
            if dis < 0.3:
                right_num += 1

        return right_num / total_len

    def qa_relevance(self, app: AppRegister, recall_res: List[List[Document]]):
        assert len(recall_res) == len(self.recall_ans)
        total_len = len(recall_res)

        revelance = 0.0

        for i in range(total_len):
            ans = self.ans[i]
            generate_res = self.generate_res[i]
            ans_embedding = app.get_embedding(ans)
            generate_res_embedding = app.get_embedding(generate_res)
            dis = cosine(ans_embedding, generate_res_embedding)
            revelance += 1 - dis

        return revelance / total_len
