# PYTHONPATH=.. python to_components.py --ds_name worldocr_v6
import os
from glob import glob
from io import BytesIO
import json
import sys
import random
import pandas as pd
from PIL import Image
from tqdm import tqdm
from to_json_doctags import parse_doctag_to_docling, _binary_hash_u64
from check_layout import infer_quality
from filtered_colored import is_colorized_overlay_page
from argparse import ArgumentParser
from docling_core.types.doc import DoclingDocument, TableItem, TextItem, GroupItem
from docling_core.transforms.serializer.markdown import MarkdownListSerializer
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams
from docling_core.types.doc.document import DOCUMENT_TOKENS_EXPORT_LABELS, DEFAULT_CONTENT_LAYERS
from docling_core.types.doc.base import BoundingBox


MARKDOWN_PROMPT = "Convert the following {tag} to {target_format}."
data_root = "../data/train2"
files = glob(f"{data_root}/*.parquet")
files = random.sample(files, k=50)

out_root = "data"
parser = ArgumentParser()
parser.add_argument("--ds_name", type=str, default="worldocr_comp_v1")
parser.add_argument("--max_samples_per_language", type=int, default=1000)
args = parser.parse_args()
ds_name = args.ds_name
max_samples_per_language = args.max_samples_per_language
out_json = f"{out_root}/{ds_name}.json"
out_images = f"{out_root}/{ds_name}_images"
out_annots = f"{out_root}/dataset_info.json"

os.makedirs(out_images, exist_ok=True)

def curate_sample(rel_image_path, content: str, prompt: str):
    return {
        "messages": [
            {"role": "user", "content": "<image>"+prompt},
            {"role": "assistant", "content": content}
        ],
        "images": [rel_image_path]
    }

data = []
append = data.append  # micro-opt

def merge_bboxes(bboxes):
    l = min(bbox.l for bbox in bboxes)
    t = min(bbox.t for bbox in bboxes)
    r = max(bbox.r for bbox in bboxes)
    b = max(bbox.b for bbox in bboxes)
    return BoundingBox(l=l, t=t, r=r, b=b)

def parse_table(item: TableItem, doc: DoclingDocument, **kwargs):
    bbox = item.prov[0].bbox
    if len(item.children) > 0: 
        childern_elements = [child.resolve(doc) for child in item.children]
        childern_bboxes = [child.prov[0].bbox for child in childern_elements if len(child.prov) > 0]
        childern_bboxes.append(bbox)
        bbox = merge_bboxes(childern_bboxes)
    return item.export_to_html(doc=doc), bbox

def parse_text(item: TextItem, **kwargs):
    return item.text, item.prov[0].bbox

def parse_group(item: GroupItem, doc: DoclingDocument, **kwargs):
    list_seralizer = MarkdownListSerializer()
    serializer = MarkdownDocSerializer(
        doc=doc,
        params=MarkdownParams(
            labels=DOCUMENT_TOKENS_EXPORT_LABELS,
            layers=DEFAULT_CONTENT_LAYERS,
            pages=None,
            start_idx=0,
            stop_idx=sys.maxsize,
            escape_html=True,
            escape_underscores=True,
            image_placeholder="<!-- image -->",
            enable_chart_tables=True,
            image_mode="placeholder",
            indent=4,
            wrap_width=None,
            page_break_placeholder=None,
            include_annotations=True,
            mark_annotations=False,
        ),
    )
    _ = serializer.serialize()
    content = list_seralizer.serialize(item=item, doc_serializer=serializer, doc=doc)
    bbox = merge_bboxes([b.item.prov[0].bbox for b in content.spans])
    content = content.text
    return content, bbox

num_samples_per_language = {}
all_dfs = []
for file in tqdm(files, desc="Loading Parquet files"):
    df = pd.read_parquet(file, columns=["id", "image", "doctag_html", "language"], engine="pyarrow")
    all_dfs.append(df)
df = pd.concat(all_dfs, ignore_index=True)
del all_dfs
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
def bbox_area(bbox):
    return (bbox.r - bbox.l) * (bbox.b - bbox.t)

# from qwen2.5-vl configs
MIN_PIXELS = 3136
MAX_PIXELS = 12845056
for row in tqdm(df.itertuples(index=False, name="Row"), desc=f"Building ...", total=len(df), leave=False):
    page_id = getattr(row, "id")
    page_image = getattr(row, "image")
    doctag_html = getattr(row, "doctag_html")
    language = getattr(row, "language")
    
    if language not in ['en', 'fr', 'de', 'es', 'it', 'pt', 'ar', 'he']:
        continue
    
    if language not in num_samples_per_language:
        num_samples_per_language[language] = 0
    if num_samples_per_language[language] > max_samples_per_language: continue
    if len(data) % 1000 == 0 and len(data) > 0: print(f"Processed sample {len(data)}", flush=True)
    try:
        image_filename = f"{page_id}.png"
        image_path = os.path.join(out_images, image_filename)

        img_bytes = page_image.get("bytes")
        pseudo_path = page_image.get("path")
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_image_path = os.path.join(tmpdirname, "temp_image")
            with open(temp_image_path, "wb") as temp_img_f:
                temp_img_f.write(img_bytes)
            if "highres" not in pseudo_path and is_colorized_overlay_page(temp_image_path):
                continue
        with Image.open(BytesIO(img_bytes)) as image:
            image_meta_data = {
                "path": pseudo_path,
                "binary_hash": _binary_hash_u64(img_bytes),
                "width": image.width,
                "height": image.height,
            }
            if "caption" not in doctag_html: continue
            doc_dict = parse_doctag_to_docling(doctag_html, image_meta_data, page_id)
            quality_score = infer_quality(image, doc_dict)
            
            if quality_score < 0.85: 
                continue
            doc = DoclingDocument.model_validate(doc_dict)
            for text_element in doc.texts:
                if "body" not in str(text_element.parent): continue
                label = text_element.label.value
                content, bbox = parse_text(text_element)
                if bbox_area(bbox) > MAX_PIXELS or bbox_area(bbox) < MIN_PIXELS: continue
                text_id = "_".join(text_element.self_ref.split("/")[1:])
                image_cropped = image.crop((bbox.l, bbox.t, bbox.r, bbox.b))
                out_image_path = os.path.join(out_images, f"{page_id}_{text_id}.png")
                image_cropped.save(out_image_path)
                rel_path = os.path.relpath(out_image_path, out_root)
                prompt = MARKDOWN_PROMPT.format(tag=label, target_format="markdown")
                append(curate_sample(rel_path, content, prompt=prompt))
                
            for table_element in doc.tables:
                label = table_element.label.value
                content, bbox = parse_table(table_element, doc)
                if bbox_area(bbox) > MAX_PIXELS or bbox_area(bbox) < MIN_PIXELS: continue
                text_id = "_".join(table_element.self_ref.split("/")[1:])
                image_cropped = image.crop((bbox.l, bbox.t, bbox.r, bbox.b))
                out_image_path = os.path.join(out_images, f"{page_id}_{text_id}.png")
                image_cropped.save(out_image_path)
                rel_path = os.path.relpath(out_image_path, out_root)
                prompt = MARKDOWN_PROMPT.format(tag=label, target_format="html")
                append(curate_sample(rel_path, content, prompt=prompt))
            
            for group_element in doc.groups:
                if group_element.content_layer.value == "furniture": continue
                label = group_element.label.value
                content, bbox = parse_group(group_element, doc)
                if bbox_area(bbox) > MAX_PIXELS or bbox_area(bbox) < MIN_PIXELS: continue
                text_id = "_".join(group_element.self_ref.split("/")[1:])
                image_cropped = image.crop((bbox.l, bbox.t, bbox.r, bbox.b))
                out_image_path = os.path.join(out_images, f"{page_id}_{text_id}.png")
                image_cropped.save(out_image_path)
                rel_path = os.path.relpath(out_image_path, out_root)
                prompt = MARKDOWN_PROMPT.format(tag=label, target_format="markdown")
                append(curate_sample(rel_path, content, prompt=prompt))
                
            with open(image_path, "wb") as img_f:
                img_f.write(img_bytes)
                
        num_samples_per_language[language] += 1

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
