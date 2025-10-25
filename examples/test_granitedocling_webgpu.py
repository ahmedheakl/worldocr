from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

# Use the original PyTorch model (not the ONNX version)
model_id = "ibm-granite/granite-3.0-vision-docling"

print("Loading model and processor...")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
print("Model loaded successfully.")

# Load the image
image_path = "data/omnidocbench_output_med/cleaned_images/doc_00b46b089d3c8267485f1ddfc49757a5617f262c_p00007.jpg"
image = Image.open(image_path)

# Prepare the prompt
prompt = "Convert this image to Docling format"

# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]
    }
]

# Apply chat template
text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

# Process inputs
inputs = processor(
    text=text,
    images=image,
    return_tensors="pt"
).to(model.device)

# Generate output
print("Generating output...")
outputs = model.generate(
    **inputs,
    max_new_tokens=4096,
    do_sample=False
)

# Decode the output
generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

# Extract only the assistant's response
if "assistant" in generated_text:
    generated_text = generated_text.split("assistant")[-1].strip()

print("\n" + "="*50)
print("Generated Docling Output:")
print("="*50)
print(generated_text)