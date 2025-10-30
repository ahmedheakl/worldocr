from PIL import Image
from vllm import LLM, SamplingParams
from tqdm import tqdm
from pathlib import Path
import torch
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
# check https://github.com/deepseek-ai/DeepSeek-OCR to download the latest vllm version


# ------------ Paths ------------
input_dir = Path('../data/omnidocbench_output_med/cleaned_images')
output_dir = Path('../data/predictions_med/deepseek-ocr')
output_dir.mkdir(parents=True, exist_ok=True)
num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
print("="*15, f"Using {num_devices} device(s) for inference.", "="*15)
# ------------ Model ------------
llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    logits_processors=[NGramPerReqLogitsProcessor]
)
sampling_param = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    # ngram logit processor args
    extra_args=dict(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822},  # whitelist: <td>, </td>
    ),
    skip_special_tokens=False,
)


# ------------ Prompt ------------
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

def is_image_file(fname: str) -> bool:
    return any(fname.lower().endswith(ext) for ext in IMAGE_EXTS)


batch_size = 8*num_devices  # adjust based on your GPU memory
valid_exts = {".png", ".jpg", ".jpeg"}
img_paths = sorted(
    [p for p in input_dir.rglob("*") if p.suffix.lower() in valid_exts]
)
for i in range(0, len(img_paths), batch_size):
    in_img_paths = img_paths[i : i + batch_size]
    batched_inputs = []
    output_stems = []  # one stem per image to name outputs

    for img_path in in_img_paths:
        with Image.open(img_path) as im:
            image = im.convert("RGB")

        
        rel = img_path.relative_to(input_dir)
        stem = rel.as_posix().rsplit(".", 1)[0].replace("/", "__")
        md_path = output_dir / f"{stem}.md"
        if md_path.exists():
            print(f"Skipping existing file: {md_path}")
            continue
        prompt = "<image>\nFree OCR."
        batched_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})
        output_stems.append(stem)
        
    outputs = llm.generate(batched_inputs, sampling_params=sampling_param)
    for stem, output, input_data in tqdm(zip(output_stems, outputs, batched_inputs), desc="Saving outputs", total=len(in_img_paths)):
        doctags = output.outputs[0].text
        md_path = output_dir / f"{stem}.md"
        doctags = doctags.replace("<think>", "").replace("</think>", "").strip()
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(doctags)

    print(f"Processed {len(in_img_paths)} images.")
        
