import os
import json
import io
import base64
import tempfile
from glob import glob
from io import BytesIO
from argparse import ArgumentParser

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from docling_core.types.doc import DoclingDocument, ImageRef

from to_json_doctags import parse_doctag_to_docling, _binary_hash_u64, to_markdown
from check_layout import infer_quality
from filtered_colored import is_colorized_overlay_page
from prompts import MARKDOWN_PROMPT, DOCTAGS_PROMPT


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--ds_name", type=str, default="worldocr_dpo_v1")
    parser.add_argument("--format", type=str, default="markdown", choices=["doctags", "markdown"])
    parser.add_argument("--max_samples_per_language", type=int, default=1000)
    parser.add_argument("--with_layouts", action="store_true")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--batch_size", type=int, default=32)  # NEW: batch size for inference
    return parser.parse_args()


def pil_image_to_data_uri(image: Image.Image, format: str = "JPEG") -> str:
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"


def curate_sample(rel_image_path, pos_content, neg_content, prompt, use_markdown_format):
    if use_markdown_format:
        pos_content = f"```markdown\n{pos_content}\n```"
    return {
        "conversations": [
            {"from": "human", "value": "<image>" + prompt},
        ],
        "images": [rel_image_path],
        "chosen": {"from": "gpt", "value": pos_content},
        "rejected": {"from": "gpt", "value": neg_content},
    }


def load_dataframes(files):
    all_dfs = []
    for file in tqdm(files, desc="Loading Parquet files"):
        df = pd.read_parquet(file, columns=["id", "image", "doctag_html", "language"], engine="pyarrow")
        all_dfs.append(df)
    df = pd.concat(all_dfs, ignore_index=True)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def initialize_llm(base_model, max_len):
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    llm = LLM(
        model=base_model,
        tensor_parallel_size=num_devices,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=max_len,
    )
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    sampling_params = SamplingParams(
        max_tokens=max_len,
        temperature=0.0,
        top_p=1.0,
    )
    return llm, processor, sampling_params


def create_batch_inference_function(llm, processor, sampling_params):
    """Create function that does batch inference"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": MARKDOWN_PROMPT},
            ],
        },
    ]
    
    def infer_batch(images):
        """Process multiple images at once"""
        if not images:
            return []
        
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = [
            {"prompt": prompt, "multi_modal_data": {"image": img}}
            for img in images
        ]
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    return infer_batch


def should_skip_sample(language, num_samples_per_language, max_samples, valid_languages):
    if language not in valid_languages:
        return True
    if language not in num_samples_per_language:
        num_samples_per_language[language] = 0
    return num_samples_per_language[language] > max_samples


def process_image_checks(img_bytes, pseudo_path):
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_image_path = os.path.join(tmpdirname, "temp_image")
        with open(temp_image_path, "wb") as temp_img_f:
            temp_img_f.write(img_bytes)
        if "highres" not in pseudo_path and is_colorized_overlay_page(temp_image_path):
            return False
    return True


def prepare_sample(row, args, num_samples_per_language):
    """Prepare sample data without inference - returns prepared data or None"""
    page_id = getattr(row, "id")
    page_image = getattr(row, "image")
    doctag_html = getattr(row, "doctag_html")
    language = getattr(row, "language")
    
    valid_languages = ['en', 'fr', 'de', 'es', 'it', 'pt', 'ar', 'he']
    if should_skip_sample(language, num_samples_per_language, args.max_samples_per_language, valid_languages):
        return None
    
    img_bytes = page_image.get("bytes")
    pseudo_path = page_image.get("path")
    
    if not process_image_checks(img_bytes, pseudo_path):
        return None
    
    image = Image.open(BytesIO(img_bytes))
    image_meta_data = {
        "path": pseudo_path,
        "binary_hash": _binary_hash_u64(img_bytes),
        "width": image.width,
        "height": image.height,
    }
    
    doc_dict = parse_doctag_to_docling(doctag_html, image_meta_data, page_id)
    if infer_quality(image, doc_dict) < 0.9:
        return None
    
    num_samples_per_language[language] += 1
    
    return {
        "page_id": page_id,
        "image": image,
        "img_bytes": img_bytes,
        "doc_dict": doc_dict,
        "language": language
    }


def finalize_sample(prepared_data, negative_content, args, out_images, out_visualizations, out_root):
    """Finalize sample after inference"""
    page_id = prepared_data["page_id"]
    image = prepared_data["image"]
    img_bytes = prepared_data["img_bytes"]
    doc_dict = prepared_data["doc_dict"]
    
    image_filename = f"{page_id}.png"
    image_path = os.path.join(out_images, image_filename)
    
    doc = DoclingDocument.model_validate(doc_dict)
    
    if args.format == "markdown":
        tags = to_markdown(doc_dict)
        prompt = MARKDOWN_PROMPT
    else:
        tags = doc.export_to_doctags()
        prompt = DOCTAGS_PROMPT
    
    with open(image_path, "wb") as img_f:
        img_f.write(img_bytes)
    
    if args.with_layouts:
        sample_image = image.convert("RGB")
        doc.pages[1].image = ImageRef(
            mimetype="image/jpeg",
            dpi=300,
            size={"width": sample_image.width, "height": sample_image.height},
            uri=pil_image_to_data_uri(sample_image, format="JPEG"),
            _pil=sample_image
        )
        imgs_by_page = doc.get_visualization(
            show_label=True,
            show_branch_numbering=False,
            viz_mode="reading_order",
            show_cell_id=True
        )
        for i, img in imgs_by_page.items():
            viz_path = os.path.join(out_visualizations, f"{doc.name}.jpg")
            img.save(viz_path)
    
    rel_path = os.path.relpath(image_path, out_root)
    return curate_sample(rel_path, tags, negative_content, prompt, args.format == "markdown")


def save_results(data, out_json, out_annots, ds_name):
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    annots = {}
    if os.path.exists(out_annots):
        with open(out_annots, "r", encoding="utf-8") as f:
            annots = json.load(f)
    
    annots[ds_name] = {
        "file_name": os.path.basename(out_json),
        "ranking": True,
        "formatting": "sharegpt",
        "columns": {
            "images": "images",
            "messages": "conversations",
            "chosen": "chosen",
            "rejected": "rejected"
        }
    }
    
    with open(out_annots, "w", encoding="utf-8") as f:
        json.dump(annots, f, indent=2)


def main():
    args = parse_arguments()
    
    data_root = "../data/train2"
    files = glob(f"{data_root}/*.parquet")
    import random
    files = random.sample(files, k=1)
    
    out_root = "data"
    out_json = f"{out_root}/{args.ds_name}.json"
    out_images = f"{out_root}/{args.ds_name}_images"
    out_visualizations = f"{out_root}/{args.ds_name}_viz"
    out_annots = f"{out_root}/dataset_info.json"
    
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_visualizations, exist_ok=True)
    
    max_len = 16384
    llm, processor, sampling_params = initialize_llm(args.base_model, max_len)
    infer_batch = create_batch_inference_function(llm, processor, sampling_params)
    
    df = load_dataframes(files)
    
    data = []
    num_samples_per_language = {}
    
    # Process in batches
    batch_prepared = []
    batch_size = args.batch_size
    
    for row in tqdm(df.itertuples(index=False, name="Row"), desc="Building dataset", total=len(df)):
        try:
            prepared = prepare_sample(row, args, num_samples_per_language)
            if prepared:
                batch_prepared.append(prepared)
            
            # When batch is full, run inference
            if len(batch_prepared) >= batch_size:
                images = [p["image"] for p in batch_prepared]
                negative_contents = infer_batch(images)
                
                for prepared, neg_content in zip(batch_prepared, negative_contents):
                    sample = finalize_sample(prepared, neg_content, args, out_images, 
                                           out_visualizations, out_root)
                    data.append(sample)
                
                if len(data) % 1000 == 0:
                    print(f"Processed {len(data)} samples", flush=True)
                
                batch_prepared = []
                
        except Exception as e:
            page_id = getattr(row, "id")
            print(f"Error processing page {page_id}: {e}", flush=True)
    
    # Process remaining batch
    if batch_prepared:
        images = [p["image"] for p in batch_prepared]
        negative_contents = infer_batch(images)
        
        for prepared, neg_content in zip(batch_prepared, negative_contents):
            sample = finalize_sample(prepared, neg_content, args, out_images, 
                                   out_visualizations, out_root)
            data.append(sample)
    
    print(f"Total processed: {len(data)} samples", flush=True)
    save_results(data, out_json, out_annots, args.ds_name)


if __name__ == "__main__":
    main()