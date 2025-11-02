from PIL import Image
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser 
from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem

parser = ArgumentParser()
parser.add_argument("--input_dir", type=str, default="../data/omnidocbench_output_med/cleaned_images")
parser.add_argument("--output_dir", type=str, default="../data/predictions_med/qwen25vl-3b-comp-v1")
args = parser.parse_args()

# ------------ Paths ------------
input_dir = Path(args.input_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

manager = InferenceManager(method="vllm")
valid_exts = {".png", ".jpg", ".jpeg"}
img_paths = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in valid_exts])
batch_size = 16
for i in tqdm(range(0, len(img_paths), batch_size), desc="Batches"):
    batch_img_paths = img_paths[i : i + batch_size]
    images = [Image.open(p).convert("RGB") for p in batch_img_paths]
    batch = [BatchInputItem(image=img, prompt_type="ocr_layout") for img in images]
    results = manager.generate(batch)
    for img_path, result in zip(batch_img_paths, results):
        rel = img_path.relative_to(input_dir)
        stem = rel.as_posix().rsplit(".", 1)[0].replace("/", "__")
        md_path = output_dir / f"{stem}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(result.markdown)

        
