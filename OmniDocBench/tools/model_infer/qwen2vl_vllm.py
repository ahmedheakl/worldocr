from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from tqdm import tqdm
from pathlib import Path

# ------------ Paths ------------
input_dir = Path('../data/omnidocbench_output_med/cleaned_images')
output_dir = Path('../data/predictions_med/qwen25vl-3b-comp-v1')
output_dir.mkdir(parents=True, exist_ok=True)
import torch
num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
print("="*15, f"Using {num_devices} device(s) for inference.", "="*15)
# ------------ Model ------------
# MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_NAME = "../checkpoints/qwen25vl-3b-comp-v1/merged_model"
# MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
max_len=8196
llm = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=num_devices,
    trust_remote_code=True,   # Qwen uses custom processors/templates
    gpu_memory_utilization=0.9,  # tweak if OOM
    max_model_len=max_len,
)
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
sampling_params = SamplingParams(
    max_tokens=max_len,        # adjust to avoid OOM
    temperature=0.0,        # deterministic
    top_p=1.0,
)


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


messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT},
        ],
    },
]

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
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        batched_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})
        output_stems.append(stem)
        
    outputs = llm.generate(batched_inputs, sampling_params=sampling_params)
    for stem, output, input_data in tqdm(zip(output_stems, outputs, batched_inputs), desc="Saving outputs", total=len(in_img_paths)):
        doctags = output.outputs[0].text
        md_path = output_dir / f"{stem}.md"
        doctags = doctags.replace("<think>", "").replace("</think>", "").strip()
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(doctags)

    print(f"Processed {len(in_img_paths)} images.")
        
