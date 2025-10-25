# PYTHONPATH=.. python to_traindata.py --ds_name worldocr_v6 --format markdown
import os
import json
from glob import glob
from io import BytesIO

import pandas as pd
from PIL import Image
from tqdm import tqdm
from docling_core.types.doc import DoclingDocument

from to_json_doctags import parse_doctag_to_docling, _binary_hash_u64, to_markdown
from argparse import ArgumentParser

DOCTAGS_PROMPT = "Convert this page to docling."
MARKDOWN_PROMPT = r'''You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

1. Text Processing:
- Accurately recognize all text content in the PDF image without guessing or inferring.
- Convert the recognized text into Markdown format.
- Maintain the original document structure, including headings, paragraphs, lists, etc.

2. Mathematical Formula Processing:
- Convert all mathematical formulas to LaTeX format.
- Enclose inline formulas with \( \). For example: This is an inline formula \( E = mc^2 \)
- Enclose block formulas with \[ \]. For example: \[ \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \]

3. Table Processing:
- Convert tables to HTML format.
- Wrap the entire table with <table> and </table>.

4. Figure Handling:
- Ignore figures content in the PDF image. Do not attempt to describe or convert images.

5. Output Format:
- Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
- For complex layouts, try to maintain the original document's structure and format as closely as possible.

Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.
'''
data_root = "../data/train2"
files = glob(f"{data_root}/*.parquet")

out_root = "data"
parser = ArgumentParser()
parser.add_argument("--ds_name", type=str, default="worldocr_v7")
parser.add_argument("--format", type=str, default="doctags", choices=["doctags", "markdown"])
parser.add_argument("--max_samples_per_language", type=int, default=1000)
args = parser.parse_args()
ds_name = args.ds_name
max_samples_per_language = args.max_samples_per_language
out_json = f"{out_root}/{ds_name}.json"
out_images = f"{out_root}/{ds_name}_images"
out_annots = f"{out_root}/dataset_info.json"

os.makedirs(out_images, exist_ok=True)

def curate_sample(rel_image_path, content, prompt=DOCTAGS_PROMPT):
    if args.format == "markdown":
        content = f"```markdown\n{content}\n```"
    return {
        "messages": [
            {"role": "user", "content": "<image>"+prompt},
            {"role": "assistant", "content": content}
        ],
        "images": [rel_image_path]
    }

data = []
append = data.append  # micro-opt


num_samples_per_language = {}
all_dfs = []
for file in tqdm(files, desc="Loading Parquet files"):
    df = pd.read_parquet(file, columns=["id", "image", "doctag_html", "language"], engine="pyarrow")
    all_dfs.append(df)
df = pd.concat(all_dfs, ignore_index=True)
del all_dfs
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# def get_components(doc):
    


for row in tqdm(df.itertuples(index=False, name="Row"), desc=f"Building ...", total=len(df), leave=False):
    page_id = getattr(row, "id")
    page_image = getattr(row, "image")
    doctag_html = getattr(row, "doctag_html")
    language = getattr(row, "language")
    
    if language not in ['en', 'fr', 'de', 'es', 'it', 'pt', 'ar', 'he']:
        continue
    
    if language not in num_samples_per_language:
        num_samples_per_language[language] = 0
    num_samples_per_language[language] += 1
    if num_samples_per_language[language] > max_samples_per_language:
        continue

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

            doc_dict = parse_doctag_to_docling(doctag_html, image_meta_data, page_id)
            if args.format == "markdown":
                tags = to_markdown(doc_dict)
                prompt = MARKDOWN_PROMPT
            else:
                doc = DoclingDocument.model_validate(doc_dict)
                tags = doc.export_to_doctags()
                prompt = DOCTAGS_PROMPT
            with open(image_path, "wb") as img_f:
                img_f.write(img_bytes)

        rel_path = os.path.relpath(image_path, out_root)
        append(curate_sample(rel_path, tags, prompt=prompt))

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
