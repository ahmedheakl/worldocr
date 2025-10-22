from transformers import AutoProcessor, AutoModelForVision2Seq
from huggingface_hub import hf_hub_download
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "ibm-granite/granite-vision-3.3-2b"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForVision2Seq.from_pretrained(model_path).to(device)

# prepare image and text prompt, using the appropriate prompt template

img_path = "/l/users/ahmed.heakl/worldocr/data/enriched_docs_charts/doc_1f8ca83325ea1a83d6ed0b24b0abe53e7913f665_p00067_0.png"

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": img_path},
            {"type": "text", "text": """Extract ONLY the chart axes from the image and return them as a two-row Markdown table.

Format EXACTLY like this (no code fences, no extra text):
| Y_AXIS_LABELS | <y_label_1> | <y_label_2> | ... |
| X_AXIS_TICKS  | <x_tick_1>  | <x_tick_2>  | ... |

Rules:
- Preserve the original spelling/script and order (Y: top→bottom; X: left→right).
- Include every category on the Y axis and every tick label on the X axis.
- If a label/tick is unreadable, write UNK.
- Do NOT add any other rows, columns, captions, bullets, or explanations.
Return ONLY the table."""},
        ],
    },
]
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(device)


# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=1024)
print(processor.decode(output[0], skip_special_tokens=True))
