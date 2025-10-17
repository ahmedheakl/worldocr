from transformers import AutoTokenizer
import json
from tqdm import tqdm

path = "LLaMA-Factory/data/worldocr_v1.json"
model = "Qwen/Qwen3-VL-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model)

def get_max_tokens(msgs) -> int:
    text = tokenizer.apply_chat_template(msgs, tokenize=False)
    tokens = tokenizer.encode(text, return_tensors="pt")
    return tokens.size(1)


with open(path, 'r') as f:
    data = json.load(f)

max_tokens = 0
for d in tqdm(data):
    messages = d['messages']
    n = get_max_tokens(messages)
    if n > 16_000:
        print("Long sample:", n)
    else:
        max_tokens = max(n, max_tokens)
    
print(f"Total tokens in sample: {max_tokens}")
