import torch
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

import random, numpy as np, torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Load images
image = load_image("data/omnidocbench_output_med/cleaned_images/doc_00b46b089d3c8267485f1ddfc49757a5617f262c_p00007.jpg")

model_name = "ibm-granite/granite-docling-258M"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    # _attn_implementation="flash_attention_2",
).to(device)

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
temperature = 0.0
inputs = processor(text=prompt, images=[image], return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=8192,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

# Generate outputs
prompt_length = inputs["input_ids"].shape[1]
trimmed_generated_ids = generated_ids[:, prompt_length:]
doctags = processor.batch_decode(
    trimmed_generated_ids,
    skip_special_tokens=False,
)[0].lstrip()

print(doctags)