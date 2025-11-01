from tqdm import tqdm
from bs4 import BeautifulSoup
from PIL import Image
import re
import os
import json
from to_json_doctags import parse_doctag_to_docling, _binary_hash_u64
from docling_core.types.doc import DoclingDocument, DocTagsDocument, ImageRef
from io import BytesIO
filter_langs = ["ru", "en", "pl", "es", "fr", "uk", "it", "sr", "hr", "bg", "ja", "cs", "ro", "de", "pt", "zh", "nl", "vi", "el", "hu", "tr"]

from glob import glob
root = "data/omnidocbench_output_large"
images_root = os.path.join(root, "images")
html_root = os.path.join(root, "htmls")
images = glob(f"{images_root}/*.jpg")
def to_html_path(image_path):
    import re
    try:
        base = re.findall(r"(doc_[a-f0-9]+_p)0*(\d+)", os.path.basename(image_path))[0]
        base = f"{base[0]}{int(base[1])}_highres"   
    except:
        # image can also be doc_out_tl_docx_batch_6_2.docx__tl__51_p00062.jpg
        # html will be doc_out_tl_docx_batch_6_2.docx__tl__51_p62_highres.html
        # tl is the language code and it can be anything 
        base = re.findall(r"(doc_.*__[^_]+__\d+_p)0*(\d+)", os.path.basename(image_path))[0]
        base = f"{base[0]}{int(base[1])}_highres"
    path = os.path.join(html_root, f"{base}.html")
    if not os.path.exists(path):
        # try without _highres
        path = image_path.replace(images_root, html_root).replace(".jpg", "_highres.html")
    return path
htmls = [to_html_path(p) for p in tqdm(images, desc="Mapping images to htmls")]

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


def get_table(element):
    data = str(element)
    doc = DocTagsDocument.from_doctags_and_image_pairs(["<doctag>"+data+"</doctag>"], [None])
    t = DoclingDocument.load_from_doctags(doc)
    html = t.export_to_html()
    soup = BeautifulSoup(html, "html.parser")
    table = str(soup.find("table"))
    return table

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

def parse_doctag(html_str, start_order=0, start_id=0, language="unknown"):
    """
    Parse <doctag> markup into OmniDocBench layout_dets entries.
    Uses BeautifulSoup to properly handle multiple/nested tags.
    """
    layout_dets = []
    order = start_order
    anno_id = start_id
    uniques_local = set()

    # Parse doctag XML
    soup = BeautifulSoup(html_str, "html.parser")
    target_elements = [f"section_header_level_{level}" for level in range(1, 10)]
    target_elements += ["text", "picture", "page_footer", "page_header", "unordered_list", "title", "otsl"]
    for element in soup.find_all(target_elements):
        if "loc_" in element.name: continue
        uniques_local.add(element.name)
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
        html = get_table(raw) if category == "table" else ""
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

        layout_dets.append(block)
        order += order_update
        anno_id += anno_id_update

    return layout_dets, order, anno_id, uniques_local


def convert_page(page_number, img_size, img_path, html_doc, language):
    width, height = img_size

    layout_dets = []
    order = 0
    anno_id = 0

    if html_doc.strip():
        dets, order, anno_id, uniques_local = parse_doctag(html_doc, start_order=order, start_id=anno_id, language=language)
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
    }, uniques_local

converted = []

with open(f"{root}/omnidocbench.json", "r", encoding="utf-8") as f:
    raw_samples = json.load(f)
    
try:
    with open(f"{root}/bad_v2.json", "r", encoding="utf-8") as f:
        bad_samples = set([os.path.basename(x).replace(".jpg", "") for x in json.load(f)])
except:
    bad_samples = set()

image_name_to_lang = {}
for d in raw_samples:
    image_name = os.path.basename(d['page_info']['image_path'])
    language = d['page_info']['page_attribute']['language']
    image_name_to_lang[image_name] = language
    
benchmark_samples = []
for image_path, html_path in zip(images, htmls):
    if not os.path.exists(html_path):
        print(f"⚠️ HTML file not found for image: {image_path}")
        continue
    page_id = os.path.basename(image_path).replace(".jpg", "")
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    language = image_name_to_lang.get(os.path.basename(image_path), "unknown")
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    benchmark_samples.append({
        "id": page_id,
        "image": {
            "path": image_path,
            "bytes": list(img_bytes)
        },
        "doctag_html": html_content,
        "language": language
    })

images_out_dir = images_root
output_dir = root
uniques = set()
for i, d in enumerate(tqdm(benchmark_samples)):
    page_id = d['id']
    if page_id in bad_samples:
        print(f"Skipping bad sample: {page_id}")
        continue
    sample_image = d['image']
    sample_html = d['doctag_html']
    sample_lang = d['language']
    img_bytes = sample_image.get("bytes")
    image = Image.open(BytesIO(bytes(img_bytes))).convert("RGB")
    image_meta_data = {
        "path": sample_image.get("path"),
        "binary_hash": _binary_hash_u64(bytes(img_bytes)),
        "width": image.width,
        "height": image.height
    }
    tags = parse_doctag_to_docling(sample_html, image_meta_data, i)
    # for t in (tags['texts'] + tags['tables'] + tags['pictures'] + tags['groups']):
    #     uniques.add(t['label'])
    doc = DoclingDocument.model_validate(tags)
    def pil_image_to_data_uri(image: Image.Image, format: str = "JPEG") -> str:
        """Convert a PIL Image to a data URI."""
        from io import BytesIO
        import base64

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
    try:
        html = doc.export_to_doctags(xsize=sample_image.width, ysize=sample_image.height)
    except Exception as e:
        print(f"❌ Error converting page {page_id} to doctags: {e}")
        continue
    page_number = page_id.split("_")[-1]
    page_number = int(re.findall(r'\d+', page_number)[0])
    new_img_path = os.path.join(images_out_dir, f"{page_id}.jpg")
    img_path = os.path.relpath(new_img_path, output_dir)
    page_obj, uniques_local = convert_page(page_number, (image.width, image.height), img_path, html, sample_lang)
    all_texts_empty = all(
        (anno['category_type'] in ['text_block', 'title', 'header', 'footer'] and not anno.get('text', '').strip()) or 
        (anno['category_type'] == 'table' and not anno.get('html', '').strip()) for anno in page_obj['layout_dets'])
    if all_texts_empty:
        print(f"Skipping page {page_id} with no text or latex.")
        continue
    # save image
    uniques.update(uniques_local)
    converted.append(page_obj)
    
print(uniques)

out_path = os.path.join(output_dir, "omnidocbench_v2.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(converted, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(converted)} pages to {out_path}")