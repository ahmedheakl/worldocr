import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from bs4 import BeautifulSoup
from PIL import Image
import re
import os
import json
import io

# --------------------------------------------
# Step 1: Load and concatenate all parquet files
# --------------------------------------------
files = glob("data/train/*.parquet")[-10:]
all_dfs = [pd.read_parquet(f) for f in tqdm(files)]
df = pd.concat(all_dfs, ignore_index=True)
filter_langs = ['en']
del all_dfs

vital_tags = {
    'equation': 10,
    'title': 2,
    'text': 1,
    'section_header_level_1': 2,
    'section_header_level_2': 2,
    'section_header_level_3': 2,
    'section_header_level_4': 2,
    'section_header_level_5': 2,
    'section_header_level_6': 2,
    'header': 3,
    'footer': 3,
    'quote': 2,
    'list': 2,
    'otsl': 5,
    'figure': 5,
    'table_caption': 5,
    'figure_caption': 5,
    'footnote': 5,
}

# TODO: 
# 1. remove table_header
# 2. remove form_tag
# 3. convert annotation to text
# 4. if the element before table_caption is a figure, convert the tag name from table_caption to figure_caption

# remove any <table_header><loc_{l1}><loc_{l2}>...<loc_{ln}></table_header> pattern from the from columns ['doctag_html', 'doctag_otsl']
# remove any <form_tag><loc_{l1}><loc_{l2}>...<loc_{ln}></form_tag> pattern from the ['doctag_html', 'doctag_otsl']
def clean_doctag_columns(doctag: str) -> str:
    import re

    # 1) remove blocks
    doctag = re.sub(r"(?is)<\s*table_header\b[^>]*>.*?</\s*table_header\s*>", "", doctag)
    doctag = re.sub(r"(?is)<\s*form_tag\b[^>]*>.*?</\s*form_tag\s*>", "", doctag)

    # 2) annotation -> text
    doctag = re.sub(r"(?i)<\s*annotation\s*>", "<text>", doctag)
    doctag = re.sub(r"(?i)</\s*annotation\s*>", "</text>", doctag)

    # 3) index structural tags (figure/table only)
    struct_tags = list(re.finditer(r"(?i)<\s*/?\s*(figure|table)\b[^>]*>", doctag))
    struct_pos = [(m.start(), m.group(1).lower()) for m in struct_tags]

    def nearest_struct_type(start_idx: int, end_idx: int):
        """Return 'figure', 'table', or None based on nearest structural tag around [start_idx, end_idx)."""
        import math
        best_type, best_dist = None, math.inf

        # nearest previous
        for pos, typ in reversed(struct_pos):
            if pos < start_idx:
                d = start_idx - pos
                best_type, best_dist = typ, d
                break

        # nearest next
        for pos, typ in struct_pos:
            if pos > end_idx:
                d = pos - end_idx
                if d < best_dist:
                    best_type, best_dist = typ, d
                break

        return best_type

    # 4) rename caption blocks based on nearest structure
    out, last = [], 0
    caption_pat = re.compile(r"(?is)<\s*table_caption\b([^>]*)>(.*?)</\s*table_caption\s*>")
    for m in caption_pat.finditer(doctag):
        s, e = m.span()
        attrs = m.group(1) or ""      # preserve any attrs if present
        inner = m.group(2)

        out.append(doctag[last:s])

        typ = nearest_struct_type(s, e)
        if typ == "figure":
            out.append(f"<figure_caption{attrs}>{inner}</figure_caption>")
        else:
            out.append(f"<table_caption{attrs}>{inner}</table_caption>")

        last = e

    out.append(doctag[last:])
    doctag = "".join(out)

    return doctag


df['doctag_html'] = df['doctag_html'].apply(clean_doctag_columns)
df['doctag_otsl'] = df['doctag_otsl'].apply(clean_doctag_columns)


def get_score(doctag_otsl: str):
    score = 0
    for tag, s in vital_tags.items():
        cnt = doctag_otsl.count(tag)
        score += cnt * s
    return score

assert {'language', 'difficulty_score'}.issubset(df.columns), "Missing columns!"

# --------------------------------------------
# Step 2: Function to sample 100 examples per language
# --------------------------------------------
def sample_normal_distribution(group: pd.DataFrame, n_samples: int = 20):
    group = group.sort_values('difficulty_score').reset_index(drop=True)
    percentiles = np.linspace(0, 100, len(group))
    group['percentile'] = percentiles
    target_bins = [0, 10, 30, 70, 90, 100]
    bin_weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1])  # approximate bell shape
    bin_counts = np.round(bin_weights * n_samples).astype(int)
    sampled = []
    for (low, high), count in zip(zip(target_bins[:-1], target_bins[1:]), bin_counts):
        candidates = group[(group['percentile'] >= low) & (group['percentile'] < high)]
        if len(candidates) > count:
            candidates = candidates.sample(count, random_state=42)
        sampled.append(candidates)
    return pd.concat(sampled)

# --------------------------------------------
# Step 3: Apply per-language sampling
# --------------------------------------------
df['difficulty_score'] = df['doctag_otsl'].apply(get_score)
benchmark_samples = df.groupby('language', group_keys=False).apply(sample_normal_distribution)
del df

benchmark_samples = benchmark_samples.drop(columns=['percentile'], errors='ignore')
print("✅ Benchmark created with", len(benchmark_samples), "samples across",
      benchmark_samples['language'].nunique(), "languages.")


# --------------------------------------------
# Step 4: Convert to OmniDocBench format and save
# --------------------------------------------

def extract_poly(text):
    """Extract bounding box from <loc_*> tags and return polygon with 4 corners."""
    coords = re.findall(r"<loc_(\d+)>", text)
    nums = [int(c) for c in coords]
    if len(nums) == 4:
        x0, y0, x1, y1 = nums
        return [x0, y0, x1, y0, x1, y1, x0, y1]  # expand bbox to polygon
    if len(nums) == 8:
        return nums  # already polygon
    return None


def clean_text(text):
    """Remove <loc_*> tags and trim whitespace."""
    return re.sub(r"<loc_\d+>", "", text).strip()

def get_table(text):
    soup = BeautifulSoup(text, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        return ""
    # The innermost/actual table is usually the last one
    return str(tables[-1])

def parse_doctag(html_str, start_order=0, start_id=0, language="unknown"):
    """
    Parse <doctag> markup into OmniDocBench layout_dets entries.
    Uses BeautifulSoup to properly handle multiple/nested tags.
    """
    layout_dets = []
    order = start_order
    anno_id = start_id

    # Parse doctag XML
    soup = BeautifulSoup(html_str, "html.parser")

    for element in soup.find_all(["text", "list", "table",
                                  "section_header_level_1",
                                  "section_header_level_2",
                                  "section_header_level_3"]):
        raw = str(element)
        poly = extract_poly(raw)
        if poly is None: continue
        inner = clean_text(element.get_text())

        if element.name.startswith("section_header_level"):
            category = "section_header"
        elif element.name == "list":
            category = "list_item"
        elif element.name == "text":
            category = "text_block"
        elif element.name == "table":
            category = "table"
        else:
            category = "text_block"

        text = inner if category not in ["table"] else ""
        html = get_table(raw) if category == "table" else ""
        block = {
            "category_type": category,
            "poly": poly,
            "ignore": False,
            "order": order,
            "anno_id": anno_id,
            "attribute": {
                "text_language": "text_english",
                "text_background": "white",
                "text_rotate": "normal"
            },
            "line_with_spans": [],
            "merge_list": []
        }
        if text:
            block["text"] = text
        if html:
            block["html"] = html

        layout_dets.append(block)
        order += 1
        anno_id += 1

    return layout_dets, order, anno_id

def convert_page(page_number, img_size, img_path, html_doc, language):
    width, height = img_size

    layout_dets = []
    order = 0
    anno_id = 0

    if html_doc.strip():
        dets, order, anno_id = parse_doctag(html_doc, start_order=order, start_id=anno_id, language=language)
        layout_dets.extend(dets)

    page_info = {
        "page_no": page_number,
        "height": height,
        "width": width,
        "image_path": img_path,
        "page_attribute": {
            "language": "english",
            "data_source": "book",
            "layout": "single_column",
            "special_issue": []
        }
    }

    extra = {"relation": []}

    return {
        "layout_dets": layout_dets,
        "page_info": page_info,
        "extra": extra
    }

output_dir = "data/omnidocbench_output_small"
os.makedirs(output_dir, exist_ok=True) 
images_out_dir = os.path.join(output_dir, "images")
os.makedirs(images_out_dir, exist_ok=True)
visualizations_out_dir = os.path.join(output_dir, "visualizations")
os.makedirs(visualizations_out_dir, exist_ok=True)
markdowns_out_dir = os.path.join(output_dir, "markdowns")
os.makedirs(markdowns_out_dir, exist_ok=True)
html_out_dir = os.path.join(output_dir, "htmls")
os.makedirs(html_out_dir, exist_ok=True)
converted = []
from to_json_doctags import parse_doctag_to_docling, _binary_hash_u64
from docling_core.types.doc import DoclingDocument, ImageRef
from io import BytesIO
for i, d in enumerate(tqdm(benchmark_samples.to_dict(orient="records"))):
    page_id = d['id']
    sample_image = d['image']
    sample_html = d['doctag_html']
    sample_lang = d['language']
    # if sample_lang not in ['fr', 'en', 'es']:
    #     continue
    img_bytes = sample_image.get("bytes")
    image = Image.open(BytesIO(bytes(img_bytes)))
    image_meta_data = {
        "path": sample_image.get("path"),
        "binary_hash": _binary_hash_u64(bytes(img_bytes)),
        "width": image.width,
        "height": image.height
    }
    tags = parse_doctag_to_docling(sample_html, image_meta_data, i)
    doc = DoclingDocument.model_validate(tags)
    def pil_image_to_data_uri(image: Image.Image, format: str = "JPEG") -> str:
        """Convert a PIL Image to a data URI."""
        from io import BytesIO
        import base64

        buffered = BytesIO()
        image.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{img_str}"
    sample_image = Image.open(io.BytesIO(sample_image['bytes'])).convert("RGB")
    doc.pages[1].image = ImageRef(
        mimetype="image/jpeg",
        dpi=300,
        size={"width": sample_image.width, "height": sample_image.height},
        uri=pil_image_to_data_uri(sample_image, format="JPEG"),
        _pil=sample_image
    )
    imgs_by_page = doc.get_visualization(
        show_label=True,               # draw label text on boxes
        show_branch_numbering=False,   # add reading-order numbering (if reading_order)
        viz_mode="reading_order",      # "reading_order" | "key_value"
        show_cell_id=True             # for key_value viz
    )
    for i, img in imgs_by_page.items():
        image_path = os.path.join(visualizations_out_dir, f"{doc.name}.jpg")
        img.save(image_path)   
        
    markdown_path = os.path.join(markdowns_out_dir, f"{doc.name}.md")
    doc.save_as_markdown(markdown_path)
    
    
    page_number = page_id.split("_")[-1]
    page_number = int(re.findall(r'\d+', page_number)[0])
    new_img_path = os.path.join(images_out_dir, f"{page_id}.jpg")
    sample_image.save(new_img_path)
    img_path = os.path.relpath(new_img_path, output_dir)
    page_obj = convert_page(page_number, (sample_image.width, sample_image.height), img_path, sample_html, sample_lang)
    converted.append(page_obj)
    
    html_path = os.path.join(html_out_dir, f"{doc.name}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(sample_html)

out_path = os.path.join(output_dir, "omnidocbench.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(converted, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(converted)} pages to {out_path}")