import sys

sys.path.append(f"{sys.path[0]}/..")

from response_gen import QwenPlusResponseGen

qwen_plus = QwenPlusResponseGen(api_key="your_api_key_here")

response = qwen_plus.response_gen("你好")

print(response)
