# Prerequisites:
# pip install vllm
# pip install docling_core
# Place your page images under ../data/omnidocbench_output_en/images

from pathlib import Path
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from PIL import Image
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from tqdm import tqdm

from typing import Any
import html

from docling_core.transforms.serializer.base import BaseDocSerializer, SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer, MarkdownParams, MarkdownTableSerializer
)
from docling_core.types.doc.document import DoclingDocument, TableItem, RichTableCell

class HtmlTableInMarkdownSerializer(MarkdownTableSerializer):
    def serialize(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        params = MarkdownParams(**kwargs)
        res_parts: list[SerializationResult] = []
        cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
        if cap_res.text:
            res_parts.append(cap_res)

        if item.self_ref in doc_serializer.get_excluded_refs(**kwargs):
            return create_ser_result(text="\n\n".join([p.text for p in res_parts]), span_source=res_parts)

        if params.include_annotations:
            ann_res = doc_serializer.serialize_annotations(item=item, **kwargs)
            if ann_res.text:
                res_parts.append(ann_res)

        def cell_text(c):
            if isinstance(c, RichTableCell):
                return doc_serializer.serialize(item=c.ref.resolve(doc=doc), **kwargs).text.replace("\n", " ")
            return (c.text or "").replace("\n", " ")

        grid = item.data.grid if item.data and item.data.grid else []

        # Split header (first row) vs body (remaining rows). If no rows, nothing to do.
        thead_rows = []
        tbody_rows = []
        if grid:
            thead_rows.append([cell_text(c) for c in grid[0]])
            for r in grid[1:]:
                tbody_rows.append([cell_text(c) for c in r])

        # Emit HTML table
        parts = ["<table>"]
        if thead_rows:
            parts.append("<thead>")
            for r in thead_rows:
                parts.append("<tr>" + "".join(f"<th>{html.escape(t)}</th>" for t in r) + "</tr>")
            parts.append("</thead>")
        parts.append("<tbody>")
        for r in tbody_rows:
            parts.append("<tr>" + "".join(f"<td>{html.escape(t)}</td>" for t in r) + "</tr>")
        parts.append("</tbody></table>")

        res_parts.append(create_ser_result(text="".join(parts), span_source=item))

        return create_ser_result(text="\n\n".join([p.text for p in res_parts if p.text]), span_source=res_parts)


# MODEL_PATH = "ibm-granite/granite-docling-258M"
MODEL_PATH = "/l/users/ahmed.heakl/worldocr/checkpoints/granitedocling2b-v2"
INPUT_DIR = Path("../data/omnidocbench_output_med/cleaned_images")
OUTPUT_DIR = Path("../data/predictions_med/docling-granite-sft")
PROMPT_TEXT = "Convert this page to docling."

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT_TEXT},
        ],
    },
]

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Collect images (recursively), deterministic order
valid_exts = {".png", ".jpg", ".jpeg"}
img_paths = sorted(
    [p for p in INPUT_DIR.rglob("*") if p.suffix.lower() in valid_exts]
)

if not img_paths:
    raise SystemExit(f"No images found in {INPUT_DIR} with extensions {sorted(valid_exts)}.")

# Initialize LLM & processor
llm = LLM(model=MODEL_PATH, revision="untied", limit_mm_per_prompt={"image": 1})
processor = AutoProcessor.from_pretrained(MODEL_PATH)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    skip_special_tokens=False,
)

# Prepare batch inputs
batch_size = 64
for i in range(0, len(img_paths), batch_size):
    in_img_paths = img_paths[i : i + batch_size]
    batched_inputs = []
    output_stems = []  # one stem per image to name outputs

    for img_path in in_img_paths:
        with Image.open(img_path) as im:
            image = im.convert("RGB")

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        batched_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})
        rel = img_path.relative_to(INPUT_DIR)
        stem = rel.as_posix().rsplit(".", 1)[0].replace("/", "__")
        output_stems.append(stem)

    outputs = llm.generate(batched_inputs, sampling_params=sampling_params)
    for stem, output, input_data in tqdm(zip(output_stems, outputs, batched_inputs), desc="Saving outputs", total=len(in_img_paths)):
        doctags = output.outputs[0].text
        md_path = OUTPUT_DIR / f"{stem}.md"
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
            [doctags],
            [input_data["multi_modal_data"]["image"]],
        )
        doc = DoclingDocument.load_from_doctags(doctags_doc, document_name=stem)
        md_serializer = MarkdownDocSerializer(
            doc=doc,
            table_serializer=HtmlTableInMarkdownSerializer(),  # <- key change
            params=MarkdownParams(
                image_placeholder="<!-- image -->",
            ),
        )
        markdown_with_html_tables = md_serializer.serialize().text
        md_path.write_text(markdown_with_html_tables, encoding="utf-8")

    print(f"Processed {len(in_img_paths)} images.")
    print(f"Markdown saved to: {OUTPUT_DIR.resolve()}")
