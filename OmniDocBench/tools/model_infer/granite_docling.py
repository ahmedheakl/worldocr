# Prerequisites:
# pip install vllm
# pip install docling_core
# Place your page images under ../data/omnidocbench_output_en/images

import time
from pathlib import Path
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from PIL import Image
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from tqdm import tqdm

# =========================
# Configuration
# =========================
MODEL_PATH = "ibm-granite/granite-docling-258M"
INPUT_DIR = Path("../data/omnidocbench_output_en/images")
OUTPUT_DIR = Path("../data/predictions/docling-granite")
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

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Collect images (recursively), deterministic order
valid_exts = {".png", ".jpg", ".jpeg"}
img_paths = sorted(
    [p for p in INPUT_DIR.rglob("*") if p.suffix.lower() in valid_exts]
)

if not img_paths:
    raise SystemExit(f"No images found in {INPUT_DIR} with extensions {sorted(valid_exts)}.")

# Initialize LLM & processor
llm = LLM(model=MODEL_PATH, revision="untied", limit_mm_per_prompt={"image": 1})
processor = AutoProcessor.from_pretrained(MODEL_PATH)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    skip_special_tokens=False,
)

# Prepare batch inputs
batched_inputs = []
output_stems = []  # one stem per image to name outputs

for img_path in img_paths:
    with Image.open(img_path) as im:
        image = im.convert("RGB")

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    batched_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})

    # Create a unique, filesystem-safe stem based on relative path
    rel = img_path.relative_to(INPUT_DIR)
    stem = rel.as_posix().rsplit(".", 1)[0].replace("/", "__")
    output_stems.append(stem)

# Run batch inference
start_time = time.time()
outputs = llm.generate(batched_inputs, sampling_params=sampling_params)

# Postprocess all results
for stem, output, input_data in tqdm(zip(output_stems, outputs, batched_inputs), desc="Saving outputs", total=len(img_paths)):
    doctags = output.outputs[0].text

    # dt_path = OUTPUT_DIR / f"{stem}.dt"
    md_path = OUTPUT_DIR / f"{stem}.md"

    # # Save raw doctags
    # dt_path.write_text(doctags, encoding="utf-8")

    # Convert doctags -> DoclingDocument -> Markdown
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
        [doctags],
        [input_data["multi_modal_data"]["image"]],
    )
    doc = DoclingDocument.load_from_doctags(doctags_doc, document_name=stem)
    doc.save_as_markdown(md_path)

print(f"Processed {len(img_paths)} images.")
print(f"Markdown saved to: {OUTPUT_DIR.resolve()}")
print(f"Total time: {time.time() - start_time:.2f} sec")
