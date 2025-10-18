# PYTHONPATH=.. python to_traindata.py
import os
import json
from glob import glob
from io import BytesIO

import pandas as pd
from PIL import Image
from tqdm import tqdm
from docling_core.types.doc import DoclingDocument

from to_json_doctags import parse_doctag_to_docling, _binary_hash_u64

PROMPT = "Convert this page to docling."
data_root = "../data/train"
files = glob(f"{data_root}/*.parquet")

out_root = "data"
ds_name = "worldocr_v2"
out_json = f"{out_root}/{ds_name}.json"
out_images = f"{out_root}/{ds_name}_images"
out_annots = f"{out_root}/dataset_info.json"

os.makedirs(out_images, exist_ok=True)

def curate_sample(rel_image_path, doctag_otsl):
    return {
        "messages": [
            {"role": "user", "content": "<image>"+PROMPT},
            {"role": "assistant", "content": doctag_otsl}
        ],
        "images": [rel_image_path]
    }

data = []
append = data.append  # micro-opt

for file in tqdm(files, desc="Files"):
    df = pd.read_parquet(file, columns=["id", "image", "doctag_html"], engine="pyarrow")
    for row in tqdm(df.itertuples(index=False, name="Row"), desc=f"[{os.path.basename(file)}]", total=len(df), leave=False):
        page_id = getattr(row, "id")
        page_image = getattr(row, "image")
        doctag_html = getattr(row, "doctag_html")

        try:
            image_filename = f"{page_id}.png"
            image_path = os.path.join(out_images, image_filename)

            img_bytes = page_image.get("bytes")
            with Image.open(BytesIO(img_bytes)) as image:
                image_meta_data = {
                    "path": page_image.get("path"),
                    "binary_hash": _binary_hash_u64(img_bytes),
                    "width": image.width,
                    "height": image.height,
                }

                print("Parsing doctags")
                doc_dict = parse_doctag_to_docling(doctag_html, image_meta_data, page_id)
                print("Validating docling document")
                doc = DoclingDocument.model_validate(doc_dict)
                print("Exporting doctags")
                tags = doc.export_to_doctags()

                with open(image_path, "wb") as img_f:
                    img_f.write(img_bytes)

            rel_path = os.path.relpath(image_path, out_root)
            append(curate_sample(rel_path, tags))

        except Exception as e:
            print(f"Error processing page {page_id} in {os.path.basename(file)}: {e}", flush=True)

    print(f"Processed sample {len(data)}", flush=True)

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

annots = {}
with open(out_annots, "r", encoding="utf-8") as f:
    annots = json.load(f)

annots[ds_name] = {
    "file_name": os.path.basename(out_json),
    "formatting": "sharegpt",
    "columns": {"messages": "messages", "images": "images"},
    "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant"
    }
}

with open(out_annots, "w", encoding="utf-8") as f:
    json.dump(annots, f, indent=2)
