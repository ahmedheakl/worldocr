from glob import glob
from tqdm import tqdm
import tempfile
import random
import re
import os
import json
from multiprocessing import Pool
import math
import base64

from docling_core.types.doc import DoclingDocument, ImageRef, DocTagsDocument
from io import BytesIO
from bs4 import BeautifulSoup
from PIL import Image
import pandas as pd
import numpy as np
import pycountry

from filtered_colored import is_colorized_overlay_page
from to_json_doctags import parse_doctag_to_docling, _binary_hash_u64, to_markdown
from check_layout import infer_quality


# --------------------------------------------
# Step 1: Load and concatenate all parquet files
# --------------------------------------------
files = glob("data/train2/*.parquet")
random.seed(42)
files = random.sample(files, k=10)  
# files = files[10:15]
all_dfs = [pd.read_parquet(f) for f in tqdm(files, desc="Loading parquet files")]
df = pd.concat(all_dfs, ignore_index=True)
filter_langs = ["ru", "en", "pl", "es", "fr", "uk", "it", "sr", "hr", "bg", "ja", "cs", "ro", "de", "pt", "zh", "nl", "vi", "el", "hu", "tr"]
del all_dfs
df = df[df['language'].isin(filter_langs)].reset_index(drop=True)
# df = df[df['doctag_html'].str.contains("caption", na=False)].reset_index(drop=True)
# df = df[df['id'].str.contains("doc_253edfcda9b18f792bd63a9ec22d9e98775fbba6_p00005")].reset_index(drop=True)
output_dir = "data/omnidocbench_output_mega"
os.makedirs(output_dir, exist_ok=True) 
images_out_dir = os.path.join(output_dir, "images")
os.makedirs(images_out_dir, exist_ok=True)
visualizations_out_dir = os.path.join(output_dir, "visualizations")
os.makedirs(visualizations_out_dir, exist_ok=True)
markdowns_out_dir = os.path.join(output_dir, "markdowns")
os.makedirs(markdowns_out_dir, exist_ok=True)
html_out_dir = os.path.join(output_dir, "htmls")
os.makedirs(html_out_dir, exist_ok=True)
doctags_out_dir = os.path.join(output_dir, "doctags")
os.makedirs(doctags_out_dir, exist_ok=True)

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

def clean_doctag_columns(doctag: str) -> str:
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
def sample_normal_distribution(group: pd.DataFrame, n_samples: int = 30):
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
# filter out samples with colorized overlay pages

def check_colorized_single(args):
    """Helper function for parallel processing."""
    idx, img_bytes, img_path = args
    if "highres" in img_path.lower():
        return None
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
        with open(tmp.name, "wb") as f:
            f.write(bytes(img_bytes))
        if is_colorized_overlay_page(tmp.name):
            return idx
    return None

def filter_colorized_overlay(df: pd.DataFrame) -> pd.DataFrame:
    # Prepare data for parallel processing
    data = [(idx, row['image']['bytes'], row['image']['path']) for idx, row in df.iterrows()]
    
    # Use multiprocessing
    with Pool(processes=4) as pool:
        results = list(tqdm(
            pool.imap(check_colorized_single, data),
            total=len(data),
            desc="Filtering colorized overlay pages"
        ))
    
    # Filter out None values to get indices to remove
    filtered_indices = [idx for idx in results if idx is not None]
    filtered_df = df.drop(index=filtered_indices).reset_index(drop=True)
    return filtered_df

def check_quality(args):
    """Helper function for parallel processing."""
    idx, doctag_html, image = args
    tags = parse_doctag_to_docling(doctag_html, {}, 0)
    image = Image.open(BytesIO(bytes(image['bytes']))).convert("RGB")
    quality = infer_quality(image, tags)
    if quality < 0.9: return idx

def filter_quality(df) -> pd.DataFrame:
    # Prepare data for parallel processing
    data = [(idx, row['doctag_html'], row['image']) for idx, row in df.iterrows()]
    
    results = []
    for args in tqdm(data, desc="Filtering low quality layout pages"):
        try:
            res = check_quality(args)
        except Exception as e:
            res = args[0]
        results.append(res)
    
    # Filter out None values to get indices to remove
    filtered_indices = [idx for idx in results if idx is not None]
    filtered_df = df.drop(index=filtered_indices).reset_index(drop=True)
    return filtered_df

df = filter_quality(df)
df = filter_colorized_overlay(df)
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
    nums = [int(c) for c in coords][:4]
    x0, y0, x1, y1 = nums
    return [x0, y0, x1, y0, x1, y1, x0, y1]  # expand bbox to polygon


def clean_text(text):
    """Remove <loc_*> tags and trim whitespace."""
    text = re.sub(r"<loc_\d+>", "", text).strip()
    text = re.sub(r"</loc_\d+>", "", text).strip()
    return text

def get_table(element):
    data = str(element)
    doc = DocTagsDocument.from_doctags_and_image_pairs(["<doctag>"+data+"</doctag>"], [None])
    t = DoclingDocument.load_from_doctags(doc)
    html = t.export_to_html()
    soup = BeautifulSoup(html, "html.parser")
    table = str(soup.find("table"))
    caption_data = None
    if "<caption>" in table:
        table = re.sub(r"<caption>.*?</caption>", "", table, flags=re.DOTALL)
        caption_match = re.search(r"<caption>(.*?)</caption>", data, re.DOTALL)
        caption_html = caption_match.group(1)
        caption_bbox = extract_poly(caption_html)
        caption_text = clean_text(caption_html)
        caption_data = (caption_text, caption_bbox)
    return table, caption_data

def get_picture_caption(element):
    data = str(element)
    caption_match = re.search(r"<caption>(.*?)</caption>", data, re.DOTALL)
    caption_html = caption_match.group(1)
    caption_bbox = extract_poly(caption_html)
    caption_text = clean_text(caption_html)
    return caption_text, caption_bbox    
        

def get_list(element, order, annot_id, attributes):
    items = []
    full_text = ""
    order_update, annot_id_update = order, annot_id
    list_ploys = []
    for item in element.find_all("list_item"):
        raw = str(item)
        poly = extract_poly(raw)
        if poly is None: continue
        list_ploys.extend(poly)
        inner = clean_text(item.get_text())
        full_text += inner + "\n"
        items.append({
            "category_type": "text_block",
            "poly": poly,
            "ignore": False,
            "order": order,
            "anno_id": annot_id,
            "attribute": attributes,
            "text": inner,
            "line_with_spans": [],
        })
        annot_id += 1
        order += 1
    order_update = order - order_update
    annot_id_update = annot_id - annot_id_update
    list_ploy = [min(list_ploys[0::2]), min(list_ploys[1::2]), max(list_ploys[0::2]), max(list_ploys[1::2])]
    list_ploy = [list_ploy[0], list_ploy[1], list_ploy[2], list_ploy[1], list_ploy[2], list_ploy[3], list_ploy[0], list_ploy[3]]
    return items, order_update, annot_id_update, full_text.strip(), list_ploy


def curate_block(category, poly, text, order, anno_id, attributes, merge_list=[], html=""):
    block = {
        "category_type": category,
        "poly": poly,
        "ignore": False,
        "order": order,
        "anno_id": anno_id,
        "attribute": attributes,
        "line_with_spans": [],
        "merge_list": merge_list
    }
    if text:
        block["text"] = text
    if html:
        block["html"] = html
    if category in ["figure"]:
        del block["line_with_spans"], block['attribute'], block['merge_list']
        if text: del block['text']
    return block

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
    target_elements = [f"section_header_level_{level}" for level in range(1, 10)]
    target_elements += ["text", "picture", "page_footer", "page_header", "unordered_list", "title", "otsl"]
    for element in soup.find_all(target_elements):
        if "loc_" in element.name: continue
        raw = str(element)
        
        attributes = {
            "text_language": f"text_{language}",
            "text_background": "white",
            "text_rotate": "normal"
        }
        if element.name == "unordered_list":
            merge_list, order_update, anno_id_update, inner, poly = get_list(element, order+1, anno_id+1, attributes)
            order_update += 1
            anno_id_update += 1
        else:
            inner = clean_text(element.get_text())
            merge_list = []
            order_update, anno_id_update = 1, 1
            poly = extract_poly(raw)
        if poly is None: continue

        if element.name == "title":
            category = "title"
        elif element.name in "page_header":
            category = "header"
        elif element.name == "page_footer":
            category = "footer"
        elif element.name == "picture":
            category = "figure"
        elif element.name in ["table", "otsl"]:
            category = "table"
        else:
            category = "text_block"

        text = inner if category not in ["table"] else ""
        html = ""
        add_later = False
        caption_data = None
        if category == "table":
            html, caption_data = get_table(raw)
            if caption_data:
                caption_text, caption_bbox = caption_data
                # check if caption_bbox is lower than table bbox, then add later
                if caption_bbox[1] >= poly[5]: add_later = True
                else:
                    caption_block = curate_block(
                        "table_caption",
                        caption_bbox,
                        caption_text,
                        order,
                        anno_id,
                        attributes
                    )
                    layout_dets.append(caption_block)
                    order += 1
                    anno_id += 1
        if category == "figure" and "<caption>" in raw:
            caption_data = get_picture_caption(raw)
            caption_text, caption_bbox = caption_data
            if caption_bbox[1] >= poly[5]: add_later = True
            else:
                caption_block = curate_block(
                    "figure_caption",
                    caption_bbox,
                    caption_text,
                    order,
                    anno_id,
                    attributes
                )
                layout_dets.append(caption_block)
                order += 1
                anno_id += 1
        block = curate_block(
            category,
            poly,
            text,
            order,
            anno_id,
            attributes,
            merge_list,
            html
        )
        layout_dets.append(block)
        order += order_update
        anno_id += anno_id_update
        
        if add_later and caption_data:
            caption_text, caption_bbox = caption_data
            caption_block = curate_block(
                f"{category}_caption",
                caption_bbox,
                caption_text,
                order,
                anno_id,
                attributes
            )
            layout_dets.append(caption_block)
            order += 1
            anno_id += 1

    return layout_dets, order, anno_id

def to_lang(code):
    language = pycountry.languages.get(alpha_2=code)
    if language: return language.name
    return "Unknown language code"



def convert_page(page_number, img_size, img_path, html_doc, language):
    language = to_lang(language).lower()
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
            "language": language,
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


def filter_tags(tags):
    to_remove = []
    for ref_id in tags['body']['children']:
        for element in tags['texts']:
            if element['self_ref'] != ref_id: continue
            if element['label'] != 'caption': continue
            cur_text = element['text']
            for text_element in tags['texts']:
                if cur_text in text_element['text']: to_remove.append(ref_id); break
            break
        
    tags['body']['children'] = [cid for cid in tags['body']['children'] if cid not in to_remove]
    tags['texts'] = [t for t in tags['texts'] if t['self_ref'] not in to_remove]
    for table in tags['tables']:
        table['children'] = [cid for cid in table['children'] if cid not in to_remove]
    for picture in tags['pictures']:
        picture['children'] = [cid for cid in picture['children'] if cid not in to_remove]
    return tags
            

converted = []
for i, d in enumerate(tqdm(benchmark_samples.to_dict(orient="records"))):
    page_id = d['id']
    sample_image = d['image']
    sample_html = d['doctag_html']
    sample_lang = d['language']
    if sample_lang not in filter_langs:
        continue
    img_bytes = sample_image.get("bytes")
    image = Image.open(BytesIO(bytes(img_bytes))).convert("RGB")
    image_meta_data = {
        "path": sample_image.get("path"),
        "binary_hash": _binary_hash_u64(bytes(img_bytes)),
        "width": image.width,
        "height": image.height
    }
    tags = parse_doctag_to_docling(sample_html, image_meta_data, i)
    tags = filter_tags(tags)
    doc = DoclingDocument.model_validate(tags)
    def pil_image_to_data_uri(image: Image.Image, format: str = "JPEG") -> str:
        """Convert a PIL Image to a data URI."""
        buffered = BytesIO()
        image.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{img_str}"
    sample_image = image
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
    try:
        html = doc.export_to_doctags(xsize=sample_image.width, ysize=sample_image.height)
    except Exception as e:
        print(f"❌ Error converting page {page_id} to doctags: {e}")
        continue
    page_number = page_id.split("_")[-1]
    page_number = int(re.findall(r'\d+', page_number)[0])
    new_img_path = os.path.join(images_out_dir, f"{page_id}.jpg")
    img_path = os.path.relpath(new_img_path, output_dir)
    page_obj = convert_page(page_number, (sample_image.width, sample_image.height), img_path, html, sample_lang)
    all_texts_empty = all(
        (anno['category_type'] in ['text_block', 'title', 'header', 'footer'] and not anno.get('text', '').strip()) or 
        (anno['category_type'] == 'table' and not anno.get('html', '').strip()) for anno in page_obj['layout_dets'])
    if all_texts_empty:
        print(f"❌ Error in page {page_id}: no text or latex.")
        continue
    for i, img in imgs_by_page.items():
        image_path = os.path.join(visualizations_out_dir, f"{page_id}.jpg")
        img.save(image_path)  
    markdown_path = os.path.join(markdowns_out_dir, f"{page_id}.md")
    markdown = to_markdown(tags)
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    sample_image.save(new_img_path)
    converted.append(page_obj)
    
    html_path = os.path.join(html_out_dir, f"{page_id}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(sample_html)
        
    doctag_path = os.path.join(doctags_out_dir, f"{page_id}.json")
    with open(doctag_path, "w", encoding="utf-8") as f:
        json.dump(tags, f, indent=2, ensure_ascii=False)

out_path = os.path.join(output_dir, "omnidocbench.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(converted, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(converted)} pages to {out_path}")