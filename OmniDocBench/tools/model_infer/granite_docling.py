# Prerequisites:
# pip install vllm
# pip install docling_core
# Place your page images under ../data/omnidocbench_output_en/images
# PYTHONPATH=.. python tools/model_infer/granite_docling.py

from pathlib import Path
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from PIL import Image
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from tqdm import tqdm
from argparse import ArgumentParser

from to_json_doctags import to_markdown

parser = ArgumentParser()
parser.add_argument("--input_dir", type=str, default="../data/omnidocbench_output_mega/images")
parser.add_argument("--output_dir", type=str, default="../data/predictions_mega/smoldocling")
parser.add_argument("--model_path", type=str, default="ds4sd/SmolDocling-256M-preview")
args = parser.parse_args()

PROMPT_TEXT = "Convert this page to docling."

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT_TEXT},
        ],
    },
]
output_dir = Path(args.output_dir)
input_dir = Path(args.input_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Collect images (recursively), deterministic order
valid_exts = {".png", ".jpg", ".jpeg"}
img_paths = sorted(
    [p for p in input_dir.rglob("*") if p.suffix.lower() in valid_exts]
)

if not img_paths:
    raise SystemExit(f"No images found in {input_dir} with extensions {sorted(valid_exts)}.")

# Initialize LLM & processor
llm = LLM(model=args.model_path, limit_mm_per_prompt={"image": 1}, gpu_memory_utilization=0.8)
processor = AutoProcessor.from_pretrained(args.model_path)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    skip_special_tokens=False,
)

# Prepare batch inputs
batch_size = 64
for i in range(0, len(img_paths), batch_size):
    in_img_paths = img_paths[i : i + batch_size]
    batched_inputs = []
    output_stems = []  # one stem per image to name outputs

    for img_path in in_img_paths:
        with Image.open(img_path) as im:
            image = im.convert("RGB")

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        batched_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})
        rel = img_path.relative_to(input_dir)
        stem = rel.as_posix().rsplit(".", 1)[0].replace("/", "__")
        output_stems.append(stem)

    outputs = llm.generate(batched_inputs, sampling_params=sampling_params)
    for stem, output, input_data in tqdm(zip(output_stems, outputs, batched_inputs), desc="Saving outputs", total=len(in_img_paths)):
        doctags = output.outputs[0].text
        md_path = output_dir / f"{stem}.md"
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
            [doctags],
            [input_data["multi_modal_data"]["image"]],
        )
        doc = DoclingDocument.load_from_doctags(doctags_doc, document_name=stem)
        markdown = to_markdown(doc.export_to_dict())
        md_path.write_text(markdown, encoding="utf-8")

    print(f"Processed {len(in_img_paths)} images.")
    print(f"Markdown saved to: {output_dir.resolve()}")
