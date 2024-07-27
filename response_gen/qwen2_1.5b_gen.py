import os
import platform
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from base import ResponseGen

DEFAULT_CKPT_PATH = 'Qwen/Qwen2-1.5B-Instruct'

class Qwen2ResponseGen(ResponseGen):
    def __init__(self, checkpoint_path=DEFAULT_CKPT_PATH, cpu_only=False, seed=1234):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() and not cpu_only else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to(self.device).eval()
        self.model.generation_config.max_new_tokens = 2048
        self.history = []
        self.seed = seed
        set_seed(self.seed)

    def response_gen(self, prompt: str) -> str:
        conversation = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
        for query_h, response_h in self.history:
            conversation.append({'role': 'user', 'content': query_h})
            conversation.append({'role': 'assistant', 'content': response_h})
        conversation.append({'role': 'user', 'content': prompt})
        inputs = self.tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors='pt').to(self.model.device)
        outputs = self.model.generate(inputs)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.history.append((prompt, response))
        print("------------")
        return response

