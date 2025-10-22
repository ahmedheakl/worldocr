# Prerequisites:
# pip install torch
# pip install docling_core
# pip install transformers

import torch
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load images
image = load_image("data/omnidocbench_output_med/images/doc_00b46b089d3c8267485f1ddfc49757a5617f262c_p00007.jpg")

# Initialize processor and model
processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
model = AutoModelForVision2Seq.from_pretrained(
    "ds4sd/SmolDocling-256M-preview",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
).to(DEVICE)

# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Convert this page to docling."}
        ]
    },
]

# Prepare inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")
inputs = inputs.to(DEVICE)

# Generate outputs
generated_ids = model.generate(**inputs, max_new_tokens=8192)
prompt_length = inputs.input_ids.shape[1]
trimmed_generated_ids = generated_ids[:, prompt_length:]
doctags = processor.batch_decode(
    trimmed_generated_ids,
    skip_special_tokens=False,
)[0].lstrip()

print(doctags)
# doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
# doc = DoclingDocument.load_from_doctags(doctags_doc, document_name="Document")

# # export as any format
# # HTML
# Path("Out/").mkdir(parents=True, exist_ok=True)
# output_path_html = Path("Out/") / "example.html"
# doc.save_as_html(output_path_html)
# # MD
# print(doc.export_to_markdown())
