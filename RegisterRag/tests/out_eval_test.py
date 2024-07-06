import sys

sys.path.append(f"{sys.path[0]}/..")

import os

from eval.our_evaluator import OurEvaluator

# data_path = os.path.join(os.path.dirname(__file__), "..", "config_template", "our_eval")
config_path = (
    "/home/yumuzhihan/Documents/Code/Project/RegisterRAG/config_template/our_eval"
)

eval = OurEvaluator(os.path.join(config_path, "config.json"))

eval.evaluate(f"{config_path}/eval_res.json")
