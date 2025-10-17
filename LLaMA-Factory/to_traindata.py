# PYTHONPATH=.. python to_traindata.py
import pandas as pd
import json
from glob import glob
from tqdm import tqdm
import os
from docling_core.types.doc import DoclingDocument

from to_json_doctags import parse_doctag_to_docling


PROMPT = "Convert this page to docling."
data_root = "../data/train"
files = glob(f"{data_root}/*.parquet")
all_df = [pd.read_parquet(f) for f in tqdm(files)]
df = pd.concat(all_df, ignore_index=True)
del all_df
print(f"Total records: {len(df)}")

def curate_sample(image_path, doctag_otsl):
    return {
        "messages": [
            {
                "role": "user",
                "content": "<image>"+PROMPT
            },
            {
                "role": "assistant",
                "content": doctag_otsl
            }
        ],
        "images": [image_path]
    }

out_root = "data"
ds_name = "worldocr_v2"
out_json = f"{out_root}/{ds_name}.json"
out_images = f"{out_root}/{ds_name}_images"
out_annots = f"{out_root}/dataset_info.json"
num_samples = 1000
os.makedirs(out_images, exist_ok=True)
data = []
# df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)
for idx, row in tqdm(df.iterrows(), total=len(df)):
    page_id = row['id']
    page_image = row['image']
    image_path = os.path.join(out_images, f"{page_id}.png")
    with open(image_path, 'wb') as f:
        f.write(page_image['bytes'])
    doc_dict = parse_doctag_to_docling(row['doctag_html'], page_image, idx)
    doc = DoclingDocument.model_validate(doc_dict)
    tags = doc.export_to_doctags()
    data.append(curate_sample(image_path.replace(f"{out_root}/", ""), tags))
    
with open(out_json, 'w') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
    
with open(out_annots, 'r') as f:
    annots = json.load(f)
    
annots[ds_name] = {
    "file_name": os.path.basename(out_json),
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  }

with open(out_annots, 'w') as f:
    json.dump(annots, f, indent=4)
