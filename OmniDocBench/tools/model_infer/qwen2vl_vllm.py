import os
import random
import numpy as np
from PIL import Image
import torch

from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# ------------ Paths ------------
input_dir = '../data/omnidocbench_output_en/images'
output_dir = '../data/predictions/qwen2vl'
os.makedirs(output_dir, exist_ok=True)

# ------------ Model ------------
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
model_max_len = 32768  # Qwen2.5 max length
llm = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=1,
    trust_remote_code=True,   # Qwen uses custom processors/templates
    dtype="float16",
    gpu_memory_utilization=0.8,  # tweak if OOM
    max_model_len=model_max_len,
)

# Chat template helper
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

# ------------ Prompt ------------
PROMPT = r'''You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

1. Text Processing:
- Accurately recognize all text content in the PDF image without guessing or inferring.
- Convert the recognized text into Markdown format.
- Maintain the original document structure, including headings, paragraphs, lists, etc.

2. Mathematical Formula Processing:
- Convert all mathematical formulas to LaTeX format.
- Enclose inline formulas with \( \). For example: This is an inline formula \( E = mc^2 \)
- Enclose block formulas with \[ \]. For example: \[ \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \]

3. Table Processing:
- Convert tables to HTML format.
- Wrap the entire table with <table> and </table>.

4. Figure Handling:
- Ignore figures content in the PDF image. Do not attempt to describe or convert images.

5. Output Format:
- Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
- For complex layouts, try to maintain the original document's structure and format as closely as possible.

Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.
'''

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

def is_image_file(fname: str) -> bool:
    return any(fname.lower().endswith(ext) for ext in IMAGE_EXTS)

def fix_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

sampling = SamplingParams(
    max_tokens=8192,        # adjust to avoid OOM
    temperature=0.0,        # deterministic
    top_p=1.0,
)

def build_prompt(image_path: str) -> str:
    """Use Qwen's chat template; put an image placeholder + the text prompt."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},  # path or URL is fine here
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

def run_one(image_path: str) -> str:
    fix_seed(0)
    prompt_text = build_prompt(image_path)
    pil_img = Image.open(image_path).convert("RGB")
    req = {
        "prompt": prompt_text,
        "multi_modal_data": {"image": [pil_img]},
    }
    outputs = llm.generate([req], sampling_params=sampling)
    return outputs[0].outputs[0].text

# ------------ Batch over directory ------------
for root, _, files in os.walk(input_dir):
    for name in files:
        if not is_image_file(name):
            continue

        image_path = os.path.join(root, name)
        basename = os.path.splitext(name)[0]
        markdown_file = os.path.join(output_dir, f"{basename}.md")

        if os.path.exists(markdown_file):
            print(f"Already here: {markdown_file}")
            continue

        try:
            md = run_one(image_path)
            with open(markdown_file, "w", encoding="utf-8") as f:
                f.write(md)
            print(f"Saved: {markdown_file}")
        except Exception as e:
            print(f"Failed on {image_path}: {e}")
