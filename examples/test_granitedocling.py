# Prerequisites:
# pip install torch
# pip install docling_core
# pip install transformers

import torch
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
image = load_image("data/omnidocbench_output_mega/images/doc_98d868836fe851387d57d08f62649a5776764c07_p00002.jpg")
model_name = "ibm-granite/granite-docling-258M"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
).to(DEVICE)

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
generated_ids = model.generate(**inputs, max_new_tokens=8192, temperature=0.0, do_sample=False)
prompt_length = inputs.input_ids.shape[1]
trimmed_generated_ids = generated_ids[:, prompt_length:]
doctags = processor.batch_decode(
    trimmed_generated_ids,
    skip_special_tokens=False,
)[0].lstrip()
print(doctags)
print("-"*20)



doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
doc = DoclingDocument.load_from_doctags(doctags_doc, document_name="Document")
html = doc.export_to_html()
print(html)
