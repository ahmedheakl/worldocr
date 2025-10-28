from docling.models.layout_model import LayoutModel
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import LayoutOptions
from PIL import Image, ImageDraw, ImageFont
import json
from glob import glob
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Load all images and annotation paths
images = glob("data/omnidocbench_output_large/images/*.jpg")
annots_paths = [im_path.replace("images", "doctags").replace(".jpg", "_highres.json") for im_path in images]

model = LayoutModel(
    artifacts_path=None,
    accelerator_options=AcceleratorOptions(),
    options=LayoutOptions(),
)

# Set batch size - adjust based on your GPU memory
# Typical values: 8-16 for 8GB GPU, 32-64 for 16GB+ GPU, 64-128 for 24GB+ GPU
batch_size = 32
iou_threshold = 0.9

os.makedirs("out_layout", exist_ok=True)
os.makedirs("best_layout", exist_ok=True)


def infer_quality(image, annots):
    result = model.layout_predictor.predict(image)
    bboxes = []
    for box in result:
        bbox = (box['l'], box['t'], box['r'], box['b'])
        label = box['label']
        confidence = box['confidence']
        bboxes.append((bbox, label, confidence))
        
    # Create visited map for model predictions
    vist_model = np.zeros((image.height, image.width), dtype=np.uint8)
    for bbox, label, confidence in bboxes:
        if label == "Page-footer": 
            continue
        l, t, r, b = bbox
        l, t, r, b = int(l), int(t), int(r), int(b)
        l = max(0, min(image.width - 1, l))
        r = max(0, min(image.width, r))
        t = max(0, min(image.height - 1, t))
        b = max(0, min(image.height, b))
        if r > l and b > t:
            vist_model[t:b, l:r] = 1
    # Create visited map for annotations
    vist_annots = np.zeros((image.height, image.width), dtype=np.uint8)
    corres_annots = annots['texts'] + annots['tables'] + annots['pictures'] + annots['groups']
    for annot in corres_annots: 
        if annot.get('label', '') == "page_footer": 
            continue
        try:
            bbox = annot['prov'][0]['bbox']
            l, t, r, b = bbox['l'], bbox['t'], bbox['r'], bbox['b']
            l, t, r, b = int(l), int(t), int(r), int(b)
            l = max(0, min(image.width - 1, l))
            r = max(0, min(image.width, r))
            t = max(0, min(image.height - 1, t))
            b = max(0, min(image.height, b))
            if r > l and b > t:
                vist_annots[t:b, l:r] = 1
        except: 
            continue
    intersection = np.logical_and(vist_model, vist_annots).sum()
    union = np.logical_or(vist_model, vist_annots).sum()
    iou = intersection / union if union > 0 else 0
    return iou

scores = []

# Process images in batches
for batch_start in tqdm(range(0, len(images), batch_size), desc="Processing batches"):
    batch_end = min(batch_start + batch_size, len(images))
    batch_paths = images[batch_start:batch_end]
    batch_annots_paths = annots_paths[batch_start:batch_end]
    
    # Load batch images and annotations
    batch_images = []
    batch_annots = []
    valid_indices = []
    valid_paths = []
    
    for idx, (path, annots_path) in enumerate(zip(batch_paths, batch_annots_paths)):
        try:
            with open(annots_path, "r", encoding="utf-8") as f:
                annot = json.load(f)
            image = Image.open(path)
            batch_images.append(image)
            batch_annots.append(annot)
            valid_indices.append(idx)
            valid_paths.append(path)
        except:
            continue
    
    if not batch_images:
        continue
    
    # BATCH PREDICTION - This is the key speedup!
    batch_results = model.layout_predictor.predict_batch(batch_images)
    
    # Process each result in the batch
    for result, image, annots, path in zip(batch_results, batch_images, batch_annots, valid_paths):
        
        # Extract bboxes from batch result
        bboxes = []
        for box in result:
            bbox = (box['l'], box['t'], box['r'], box['b'])
            label = box['label']
            confidence = box['confidence']
            bboxes.append((bbox, label, confidence))
        
        # Create visited map for model predictions using numpy for faster IOU calculation
        vist_model_np = np.zeros((image.height, image.width), dtype=np.uint8)
        for bbox, label, confidence in bboxes:
            if label == "Page-footer": 
                continue
            l, t, r, b = bbox
            l, t, r, b = int(l), int(t), int(r), int(b)
            l = max(0, min(image.width - 1, l))
            r = max(0, min(image.width, r))
            t = max(0, min(image.height - 1, t))
            b = max(0, min(image.height, b))
            if r > l and b > t:
                vist_model_np[t:b, l:r] = 1
        
        # Create visited map for annotations
        vist_annots_np = np.zeros((image.height, image.width), dtype=np.uint8)
        corres_annots = annots['texts'] + annots['tables'] + annots['pictures'] + annots['groups']
        for annot in corres_annots:
            if annot.get('label', '') == "page_footer": 
                continue
            try:
                bbox = annot['prov'][0]['bbox']
                l, t, r, b = bbox['l'], bbox['t'], bbox['r'], bbox['b']
                l, t, r, b = int(l), int(t), int(r), int(b)
                l = max(0, min(image.width - 1, l))
                r = max(0, min(image.width, r))
                t = max(0, min(image.height - 1, t))
                b = max(0, min(image.height, b))
                if r > l and b > t:
                    vist_annots_np[t:b, l:r] = 1
            except: 
                continue
        
        # Calculate IOU using numpy - MUCH faster than pixel-by-pixel iteration
        intersection = np.logical_and(vist_model_np, vist_annots_np).sum()
        union = np.logical_or(vist_model_np, vist_annots_np).sum()
        iou = intersection / union if union > 0 else 0
        scores.append(iou)
        print(f"IOU for {os.path.basename(path)}: {iou:.4f}")
        viz_path = path.replace("images", "visualizations").replace(".jpg", "_highres.jpg")
        viz = Image.open(viz_path)
        if iou >= iou_threshold:
            # Save best layouts
            outname = os.path.basename(viz_path)
            viz.save(f"best_layout/{outname}")
        
        # Create visualization
        # outname = os.path.basename(path)
        # united_image = Image.new("RGB", (image.width * 3, image.height))
        
        
        # try:
        #     viz = Image.open(viz_path)
        #     united_image.paste(viz, (0, 0))
        # except:
        #     # If visualization doesn't exist, use original image
        #     united_image.paste(image, (0, 0))
        
        # # Convert numpy arrays back to PIL for visualization
        # vist_model_pil = Image.fromarray(vist_model_np * 255, mode='L').convert("RGB")
        # vist_annots_pil = Image.fromarray(vist_annots_np * 255, mode='L').convert("RGB")
        
        # united_image.paste(vist_model_pil, (image.width, 0))
        # united_image.paste(vist_annots_pil, (image.width * 2, 0))
        # draw = ImageDraw.Draw(united_image)
        
        # font = ImageFont.load_default(size=50)
        # draw.text((image.width * 1.5, 10), f"IOU: {iou:.4f}", fill="white", align="center", font=font)
        
        # united_image.save(f"out_layout/{outname}_united.png")
        
# After processing all batches, plot the IOU score distribution
plt.figure(figsize=(10, 6))
plt.hist(scores, bins=50, color='blue', alpha=0.7)
plt.title("Distribution of Layout Quality Scores (IOU)")
plt.xlabel("IOU Score")
plt.ylabel("Frequency")
plt.savefig("layout_quality_scores_distribution.png")
