from PIL import Image
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser 
import torch

import os
import re
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'

from deepseek_ocr.config import MODEL_PATH, PROMPT, MAX_CONCURRENCY, CROP_MODE, NUM_WORKERS
from concurrent.futures import ThreadPoolExecutor
from deepseek_ocr.deepseek_ocr import DeepseekOCRForCausalLM
from vllm.model_executor.models.registry import ModelRegistry
from vllm import LLM, SamplingParams
from deepseek_ocr.process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from deepseek_ocr.process.image_process import DeepseekOCRProcessor
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

parser = ArgumentParser()
parser.add_argument("--input_dir", type=str, default="../data/omnidocbench_output_med/cleaned_images")
parser.add_argument("--output_dir", type=str, default="../data/predictions_med/qwen25vl-3b-comp-v1")
args = parser.parse_args()

# ------------ Paths ------------
input_dir = Path(args.input_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
# ------------ Model ------------
max_len=8196
llm = LLM(
    model=MODEL_PATH,
    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
    block_size=256,
    enforce_eager=False,
    trust_remote_code=True, 
    max_model_len=8192,
    swap_space=0,
    max_num_seqs = MAX_CONCURRENCY,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
)
logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=40, window_size=90, whitelist_token_ids= {128821, 128822})] #window for fast；whitelist_token_ids: <td>,</td>
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    logits_processors=logits_processors,
    skip_special_tokens=False,
)

class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    RESET = '\033[0m' 

def clean_formula(text):
    formula_pattern = r'\\\[(.*?)\\\]'
    def process_formula(match):
        formula = match.group(1)
        formula = re.sub(r'\\quad\s*\([^)]*\)', '', formula)
        formula = formula.strip()
        return r'\[' + formula + r'\]'
    cleaned_text = re.sub(formula_pattern, process_formula, text)
    return cleaned_text

def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    mathes_other = []
    for a_match in matches:
        mathes_other.append(a_match[0])
    return matches, mathes_other

def process_single_image(image):
    prompt_in = prompt
    cache_item = {
        "prompt": prompt_in,
        "multi_modal_data": {"image": DeepseekOCRProcessor().tokenize_with_images(images = [image], bos=True, eos=True, cropping=CROP_MODE)},
    }
    return cache_item


if __name__ == "__main__":
    valid_exts = {".png", ".jpg", ".jpeg"}
    img_paths = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in valid_exts])
    prompt = PROMPT
    images = [Image.open(p).convert('RGB') for p in img_paths]
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:  
        batch_inputs = list(tqdm(
            executor.map(process_single_image, images),
            total=len(images),
            desc="Pre-processed images"
        ))
    outputs_list = llm.generate(
        batch_inputs,
        sampling_params=sampling_params
    )
    for output, image in zip(outputs_list, img_paths):
        image = str(image)
        content = output.outputs[0].text
        mmd_det_path = args.output_dir + image.split('/')[-1].replace('.jpg', '_det.md')
        with open(mmd_det_path, 'w', encoding='utf-8') as afile:
            afile.write(content)
        content = clean_formula(content)
        matches_ref, mathes_other = re_match(content)
        for idx, a_match_other in enumerate(tqdm(mathes_other, desc="other")):
            content = content.replace(a_match_other, '').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n').replace('<center>', '').replace('</center>', '')
        mmd_path = args.output_dir + image.split('/')[-1].replace('.jpg', '.md')
        with open(mmd_path, 'w', encoding='utf-8') as afile:
            afile.write(content)

            
