from datasets import load_dataset
from tqdm import tqdm
from to_json_doctags import parse_doctag_to_docling, _binary_hash_u64
from PIL import Image
from docling_core.types.doc import (
    DoclingDocument,
    ImageRef
)
from to_json_doctags import to_markdown
import os

# ds_id = "youssefkhalil320/urdu_images_doc_tags_all_v9"
# ds_id = "ahmedheakl/wordocr_instruct_v4"
ds_id = "youssefkhalil320/persian_images_doc_tags_all_v12"
split="train"
ds = load_dataset(ds_id, split=split)
# sample 100 samples
ds = ds.shuffle(seed=42).select(range(100))
visualizations_out_dir = f"data/validated_visualizations_khalil_persian"
os.makedirs(visualizations_out_dir, exist_ok=True)

for idx, row in enumerate(tqdm(ds)):
    # if row['has_non_full_width_text']: continue
    image = row["image"]
    image_path = row['id'].replace(".pdf", ".jpg")
    img_bytes = image.tobytes()
    image_meta_data = {
        "path": image_path,
        "binary_hash": _binary_hash_u64(bytes(img_bytes)),
        "width": image.width,
        "height": image.height
    }
    doc_dict = parse_doctag_to_docling(row["doctag_html"], image_meta_data, 0)
    markdown = to_markdown(doc_dict)
    doc = DoclingDocument.model_validate(doc_dict)
    def pil_image_to_data_uri(image: Image.Image, format: str = "JPEG") -> str:
        """Convert a PIL Image to a data URI."""
        from io import BytesIO
        import base64

        buffered = BytesIO()
        image.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{img_str}"
    sample_image = image.convert("RGB")
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
        image_path = os.path.join(visualizations_out_dir, f"{doc.name}_{idx}.jpg")
        img.save(image_path)   
    with open(os.path.join(visualizations_out_dir, f"{doc.name}_{idx}.md"), "w", encoding="utf-8") as f:
        f.write(row["doctag_html"])
        