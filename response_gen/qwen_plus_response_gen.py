import requests

import dashscope
import time
from http import HTTPStatus

from .base import ResponseGen


class QwenPlusResponseGen(ResponseGen):
    def __init__(self, api_key: str) -> None:
        super().__init__()
        self.api_key = api_key
        self.retry = 0
        return

    def response_gen(self, prompt: str) -> str:
        dashscope.api_key = self.api_key
        responses = dashscope.Generation.call(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            result_format="message",
            stream=False,
        )
        if "output" not in responses:
            self.retry += 1
            if self.retry >= 3:
                return "Error: No response from Qwen-plus"
            time.sleep(5)
            return self.response_gen(prompt)
        if responses.status_code == HTTPStatus.OK:
            output = responses["output"]["choices"][0]["message"]["content"]
            self.retry = 0
            return output
        else:
            self.retry += 1
            if self.retry >= 3:
                return "Error: No response from Qwen-plus"
            time.sleep(5)
            return self.response_gen(prompt)
