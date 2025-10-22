import requests
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
import torch
from glob import glob
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont


classes_map = {
    0: "Caption",
    1: "Footnote",
    2: "Formula",
    3: "List-item",
    4: "Page-footer",
    5: "Page-header",
    6: "Picture",
    7: "Section-header",
    8: "Table",
    9: "Text",
    10: "Title",
    11: "Document Index",
    12: "Code",
    13: "Checkbox-Selected",
    14: "Checkbox-Unselected",
    15: "Form",
    16: "Key-Value Region",
}
model_name = "ds4sd/docling-layout-heron"
threshold = 0.6

out_dir = "data/rtdetr_outputs"


# Initialize the model
image_processor = RTDetrImageProcessor.from_pretrained(model_name)
model = RTDetrV2ForObjectDetection.from_pretrained(model_name)

# Run the prediction pipeline
def infer(image_path):
    image = Image.open(image_path)
    image = image.convert("RGB")
    inputs = image_processor(images=[image], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    results = image_processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([image.size[::-1]]),
        threshold=threshold,
    )

    # draw the results on the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for result in results:
        for score, label_id, box in zip(
            result["scores"], result["labels"], result["boxes"]
        ):
            score = round(score.item(), 2)
            label = classes_map[label_id.item()]
            box = [round(i, 2) for i in box.tolist()]
            draw.rectangle(box, outline="blue", width=4)
            draw.text((box[0], box[1]), f"{label}", fill="red", font=font)
            
    image.save("output_with_detections.jpg")
    image.close()
            
imgs = glob("data/omnidocbench_output_med/images/*.jpg")            
for img_path in tqdm(imgs):
    infer(img_path)
     
        
        
# Save or display the image

        
        
