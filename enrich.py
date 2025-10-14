import os, multiprocessing as mp
import time
import json
from pathlib import Path
from docling.datamodel.document import InputDocument, InputFormat
from docling_ibm_models.document_figure_classifier_model.document_figure_classifier_predictor import (
    DocumentFigureClassifierPredictor,
)
from huggingface_hub import snapshot_download
from docling_core.types.doc import BoundingBox
import re
from typing import Optional, Tuple, Any, List, Dict
from docling_core.types.doc import (
    CodeItem,
    DoclingDocument
)
from docling_core.types.doc.labels import CodeLanguageLabel, DocItemLabel
from collections import defaultdict
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from PIL import Image
import fitz  # PyMuPDF


def get_available_gpu_ids_from_env() -> List[int]:
    """Parses CUDA_VISIBLE_DEVICES and returns list of actual GPU IDs."""
    env_val = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not env_val:
        raise RuntimeError("CUDA_VISIBLE_DEVICES is not set")
    return [int(g.strip()) for g in env_val.split(",") if g.strip().isdigit()]


INPUT_PATH = Path(
    "data/docling_json_out_validated"
)
OUTPUT_PATH = Path(
    "data/enriched_docs"
)


def get_code_formula_model():
    llm = LLM(
        model="ds4sd/CodeFormulaV2",
        limit_mm_per_prompt={"image": 1},
        seed=42,
        gpu_memory_utilization=0.5
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        skip_special_tokens=False,
    )
    processor = AutoProcessor.from_pretrained("ds4sd/CodeFormulaV2")
    print("✅ Loaded CodeFormula model.")
    return llm, processor, sampling_params


def get_document_picture_classifier(device_id) -> DocumentFigureClassifierPredictor:
    download_path = snapshot_download(
        repo_id="ds4sd/DocumentFigureClassifier",
        force_download=False,
        revision="v1.0.1",
    )
    download_path = Path(download_path)
    device = f"cuda:{device_id}"
    return DocumentFigureClassifierPredictor(
        artifacts_path=download_path, device=device
    )


def get_backend(p: Path):
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    in_doc = InputDocument(
        path_or_stream=p,
        filename="file.pdf",
        format=InputFormat.PDF,
        backend=PyPdfiumDocumentBackend,
    )
    if in_doc.valid:
        try:
            return in_doc._backend
        except AttributeError:
            return None
    return None


def extract_code_language(input_string: str) -> Tuple[str, Optional[str]]:
    """Extracts a programming language from the beginning of a string.

    This function checks if the input string starts with a pattern of the form
    ``<_some_language_>``. If it does, it extracts the language string and returns
    a tuple of (remainder, language). Otherwise, it returns the original string
    and `None`.

    Args:
        input_string (str): The input string, which may start with ``<_language_>``.

    Returns:
        Tuple[str, Optional[str]]:
            A tuple where:
            - The first element is either:
                - The remainder of the string (everything after ``<_language_>``),
                if a match is found; or
                - The original string, if no match is found.
            - The second element is the extracted language if a match is found;
            otherwise, `None`.
    """
    pattern = r"^<_([^>]+)_>\s*(.*)"
    match = re.match(pattern, input_string, flags=re.DOTALL)
    if not match: return input_string, None
    language = str(match.group(1))  # the captured programming language
    remainder = str(match.group(2))  # everything after the <_language_>
    return remainder, language
        


def get_code_language_enum(value: Optional[str]) -> CodeLanguageLabel:
    """
    Converts a string to a corresponding `CodeLanguageLabel` enum member.

    If the provided string does not match any value in `CodeLanguageLabel`,
    it defaults to `CodeLanguageLabel.UNKNOWN`.

    Args:
        value (Optional[str]): The string representation of the code language or None.

    Returns:
        CodeLanguageLabel: The corresponding enum member if the value is valid,
        otherwise `CodeLanguageLabel.UNKNOWN`.
    """
    if not isinstance(value, str):
        return CodeLanguageLabel.UNKNOWN

    try:
        return CodeLanguageLabel(value)
    except ValueError:
        return CodeLanguageLabel.UNKNOWN


def expand_bbox(bbox, expansion_factor=0.05):
    width = bbox.r - bbox.l
    height = bbox.b - bbox.t

    return BoundingBox(
        l=bbox.l - width * expansion_factor,
        t=bbox.t - height * expansion_factor,
        r=bbox.r + width * expansion_factor,
        b=bbox.b + height * expansion_factor,
        coord_origin=bbox.coord_origin,
    )

def get_images(items: Any, json_path: str):
    json_name = os.path.basename(json_path)
    image_path = os.path.join("data/docling_images_out", json_name.replace(".json", ".jpg"))
    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    images = []
    labels = []
    
    for text_item in items:
        label = text_item.label
        bbox = text_item.prov[0].bbox
        bbox = expand_bbox(bbox)
        left  = max(0, int(bbox.l))
        upper = max(0, int(bbox.t))   # top
        right = min(W, int(bbox.r))
        lower = min(H, int(bbox.b))   # bottom
        cropped_image = image.crop((left, upper, right, lower))
        images.append(cropped_image)
        labels.append(str(label))
        
    return images, labels
    


def get_images_pdf(items: Any, scale: float, doc_backend):
    images = []
    labels = []
    previous_page_no = None
    page_backend = None

    for text_item in items:
        label = text_item.label

        page_no = text_item.prov[0].page_no
        bbox = text_item.prov[0].bbox
        bbox = expand_bbox(bbox)

        if page_no != previous_page_no:
            if page_backend is not None:
                page_backend.unload()
            page_backend = doc_backend.load_page(page_no - 1)
            previous_page_no = page_no

        page_image = page_backend.get_page_image(scale=scale, cropbox=bbox)
        images.append(page_image)
        labels.append(str(label))

    if page_backend is not None:
        page_backend.unload()
    return images, labels


def get_batch_of_images(folders):
    images = []
    doc_ids = []
    labels = []
    for folder in folders:
        try:
            json_path = INPUT_PATH / folder
            with open(json_path) as f:
                doc_dict = json.load(f)
            doc = DoclingDocument.model_validate(doc_dict)
            texts = [
                t
                for t in doc.texts
                if t.label == DocItemLabel.CODE or t.label == DocItemLabel.FORMULA
            ]
            pictures = [
                t
                for t in doc.pictures
            ]
            
            if not texts and not pictures:
                continue
            items = texts + pictures
            images_doc, labels_doc = get_images(items, json_path)
            doc_ids.extend([folder] * len(images_doc))
            labels.extend(labels_doc)
            images.extend(images_doc)
            for i, img in enumerate(images_doc):
                name = os.path.splitext(os.path.basename(json_path))[0]
                img.save(OUTPUT_PATH / f"{name}_{i}.png")
            print(f"📖 Processing document {folder} with {len(doc.texts)} text items", flush=True)
            if len(images) >= BATCH_SIZE:
                yield images, labels, doc_ids
                images = []
                doc_ids = []
                labels = []
        except Exception as e:
            print(f"❌ Error during processing {folder}, {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue
    
    if len(images) > 0:
        yield images, labels, doc_ids
        print("📖 Reader exiting", flush=True)


def partition_by_id(l: List[str], doc_ids: List[str]) -> Dict[int, List[str]]:
    partitioned = defaultdict(list)
    for el, doc_id in zip(l, doc_ids):
        partitioned[doc_id].append(el)
    return partitioned


def sanitize(outputs):
    to_replace = ["<loc_0><loc_0><loc_500><loc_500>", "<formula>", "</formula>", "<code>", "</code>", "<|pad|>", "<end_of_utterance>"]
    sanitized_output = []
    for output in outputs:
        if not isinstance(output, str): 
            sanitized_output.append(output)
            continue
        output = output.strip()
        for rpl in to_replace:
            output = output.replace(rpl, "")
        sanitized_output.append(output)

    return sanitized_output

def pdf_to_images(pdf_path: Path, output_dir: Path) -> None:
    """
    Converts each page of the PDF at `pdf_path` to a PNG image at 140 DPI
    and saves them in `output_dir`.

    Args:
        pdf_path (Path): Path to the input PDF file.
        output_dir (Path): Directory where output images will be saved.
    """
    scale = 140 / 72.0
    zoom_matrix = fitz.Matrix(scale, scale)

    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=zoom_matrix, alpha=False)
        out_path = output_dir / f"image_{i}.png"
        pix.save(out_path)

def update_docs(model_outputs, doc_ids):
    model_outputs = sanitize(model_outputs)
    d = partition_by_id(model_outputs, doc_ids)
    for folder, els in d.items():
        json_path = INPUT_PATH / folder
        enriched_js = OUTPUT_PATH / folder

        try:
            # ----- load doc ----------------------------------------------------------
            with open(json_path) as f:
                doc_dict = json.load(f)
            doc = DoclingDocument.model_validate(doc_dict)

            # ----- pick items & backend ---------------------------------------------
            texts = [
                t
                for t in doc.texts
                if t.label == DocItemLabel.CODE or t.label == DocItemLabel.FORMULA
            ]
            pictures = [
                t
                for t in doc.pictures
            ]
            items = texts + pictures
            assert len(els) == len(items)
            for t, o in zip(items, els):
                if isinstance(o, list):
                    classes = [{"class_name": cls[0], "confidence": float(cls[1])} for cls in o]
                    t.annotations.append({"kind": "classification", "provenance": "model:DocumentFigureClassifier", "predicted_classes": classes})
                    continue
                o, lang = extract_code_language(o)
                if isinstance(t, CodeItem):
                    t.code_language = get_code_language_enum(lang)
                t.text = o

            doc.save_as_json(enriched_js)

            doctags = (
                doc.export_to_doctags()
                .replace("<doctag>", "")
                .replace("</doctag>", "")
                .split("<page_break>")
            )
            doctags = [f"<doctag>{dt.strip()}\n</doctag>" for dt in doctags]

            for page_idx, doctag in enumerate(doctags, start=1):
                name = os.path.basename(json_path).replace(".json", "")
                with open(OUTPUT_PATH / f"{name}_dt_{page_idx}.dt", "w") as fp:
                    fp.write(doctag)
            print(f"✅ Document {folder} done", flush=True)
        except Exception as e:
            print(f"❌ Could not update folder: {folder}, {e}", flush=True)
            import traceback    
            traceback.print_exc()
            continue


AVAILABLE_GPUS = get_available_gpu_ids_from_env()
N_GPUS = len(AVAILABLE_GPUS)
print(f"GPUs visible: {AVAILABLE_GPUS}", flush=True)
BATCH_SIZE = 160
PREFETCH_FACTOR = 100  # batches kept ready per GPU
SENTINEL = object()  # unique end‑of‑stream marker
BATCH_QUEUE_SIZE = 2 * N_GPUS  # BATCH_SIZE * PREFETCH_FACTOR * N_GPUS
RESULT_QUEUE_SIZE = 0 # unlimited


def batch_producer(folders: list[str], batch_q: mp.Queue, device_count: int) -> None:
    """
    Continuously fills `batch_q` with (images, labels, doc_ids) tuples.
    No CUDA code – safe on CPU‑only node.
    """
    try:
        for images, labels, doc_ids in get_batch_of_images(folders):
            print(
                f"🚜 Produced {len(images)} images for documents",
                flush=True,
            )
            batch_q.put((images, labels, doc_ids))
    finally:
        print("☢️ Reader putting sentinels.")
        # one sentinel for each GPU worker
        for _ in range(device_count):
            batch_q.put(SENTINEL)


def get_prompt(user_message, processor):
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": user_message}],
        },
    ]
    
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    return prompt


def predict(model, sampling_params, image_batch, prompts):
    
    llm_inputs = [
            {"prompt": prompt, "multi_modal_data": {"image": img}}
            for img, prompt in zip(image_batch, prompts)
        ]
    outputs = model.generate(llm_inputs, sampling_params=sampling_params)
    outputs = [output.outputs[0].text for output in outputs]
    
    return outputs

def gpu_worker(
    local_rank: int, global_gpu_id: int, batch_q: mp.Queue, result_q: mp.Queue
) -> None:

    os.environ["CUDA_VISIBLE_DEVICES"] = str(global_gpu_id)
    import torch

    model, processor, sampling_params = get_code_formula_model()
    figure_model = get_document_picture_classifier(global_gpu_id)
    code_prompt = get_prompt("<code>", processor)
    formula_prompt = get_prompt("<formula>", processor)
    while True:
        t0 = time.time()
        batch = batch_q.get()
        t1 = time.time()
        if batch is SENTINEL:
            result_q.put(SENTINEL)  # pass downstream
            break
        wait_time = t1 - t0
        print(f"🧑‍🚀 GPU worker {local_rank} on GPU {global_gpu_id} ready", flush=True)
        preds = []
        images, labels, doc_ids = batch
        for i in range(0, len(images), BATCH_SIZE):
            image_batch = images[i : i + BATCH_SIZE]
            label_batch = labels[i : i + BATCH_SIZE]
            preds_batch = []

            prompt_map = {"code": code_prompt, "formula": formula_prompt}
            text_indices = [idx for idx, label in enumerate(label_batch) if label in prompt_map]
            figures_indices = [idx for idx, label in enumerate(label_batch) if label in "picture"]
            text_images = [image_batch[idx] for idx in text_indices]
            text_prompts = [prompt_map[label] for idx, label in enumerate(label_batch) if idx in text_indices]
            text_preds = predict(model, sampling_params, text_images, text_prompts)
            
            figures = [image_batch[idx] for idx in figures_indices]
            figure_preds = figure_model.predict(figures)
            figure_preds = [pred[:3] for pred in figure_preds]
            
            for idx in range(len(image_batch)):
                if idx in text_indices:
                    pred = text_preds[text_indices.index(idx)]
                    preds_batch.append(pred)
                elif idx in figures_indices:
                    pred = figure_preds[figures_indices.index(idx)]
                    preds_batch.append(pred)
                else:
                    raise ValueError(f"Unknown label {label_batch[idx]} at index {idx}")
            preds.extend(preds_batch)

        print(
            f"🚀 [GPU {global_gpu_id}] preds={len(preds)}, images={len(images)}, wait_time={wait_time:.2f}s, doc_ids={set(doc_ids)}",
            flush=True,
        )
        result_q.put((preds, doc_ids))

    del model
    torch.cuda.empty_cache()


def doc_writer(result_q: mp.Queue, remaining_sentinels: int) -> None:
    """
    Consumes inference results and writes back enriched JSON.
    """
    finished = 0
    while finished < remaining_sentinels:
        item = result_q.get()
        if item is SENTINEL:
            finished += 1
            continue

        preds, doc_ids = item
        update_docs(preds, doc_ids)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    if N_GPUS == 0:
        raise RuntimeError("No GPU found")

    if os.path.exists(OUTPUT_PATH):
        ALREADY_PROCESSED = set(os.listdir(OUTPUT_PATH))
    else:
        ALREADY_PROCESSED = set()
    DOC_FOLDERS = sorted(os.listdir(INPUT_PATH))
    DOC_FOLDERS = [f for f in DOC_FOLDERS if f not in ALREADY_PROCESSED]

    # ------------- queues ----------------------------------------------------
    batch_q = mp.Queue(maxsize=BATCH_QUEUE_SIZE)
    result_q = mp.Queue(maxsize=RESULT_QUEUE_SIZE)

    # ------------- processes -------------------------------------------------
    procs: list[mp.Process] = []

    # 0. producer (single CPU process)
    producer = mp.Process(
        target=batch_producer, args=(DOC_FOLDERS, batch_q, N_GPUS), daemon=False
    )
    producer.start()
    procs.append(producer)

    # 1. GPU workers
    for local_rank, gpu_id in enumerate(AVAILABLE_GPUS):
        p = mp.Process(
            target=gpu_worker,
            args=(local_rank, gpu_id, batch_q, result_q),
            daemon=False,
        )
        p.start()
        procs.append(p)

    # 2. writer (single CPU process)
    writer = mp.Process(target=doc_writer, args=(result_q, N_GPUS), daemon=False)
    writer.start()
    procs.append(writer)

    for p in procs:
        p.join()

    print("✅ pipeline completed", flush=True)