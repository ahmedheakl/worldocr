# Prerequisites:
# pip install torch
# pip install docling_core
# pip install transformers

import torch
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load images
image = load_image("data/omnidocbench_output_med/cleaned_images/doc_00b46b089d3c8267485f1ddfc49757a5617f262c_p00007.jpg")

model_name = "ibm-granite/granite-docling-258M"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2",
).to(DEVICE)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Convert this page to docling."}
        ]
    },
]

# Prepare inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")
inputs = inputs.to(DEVICE)

# Generate outputs
generated_ids = model.generate(**inputs, max_new_tokens=8192, temperature=0.0, do_sample=False)
prompt_length = inputs.input_ids.shape[1]
trimmed_generated_ids = generated_ids[:, prompt_length:]
doctags = processor.batch_decode(
    trimmed_generated_ids,
    skip_special_tokens=False,
)[0].lstrip()
print(doctags)
print("-"*20)



doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
doc = DoclingDocument.load_from_doctags(doctags_doc, document_name="Document")
html = doc.export_to_html()
print(html)
# exit()
# from pathlib import Path
# from typing import Any, Optional
# import html

# from docling_core.transforms.serializer.base import BaseDocSerializer, SerializationResult
# from docling_core.transforms.serializer.common import create_ser_result
# from docling_core.transforms.serializer.markdown import (
#     MarkdownDocSerializer, MarkdownParams, MarkdownTableSerializer
# )
# from docling_core.types.doc.document import DoclingDocument, TableItem, RichTableCell

# class HtmlTableInMarkdownSerializer(MarkdownTableSerializer):
#     def serialize(
#         self,
#         *,
#         item: TableItem,
#         doc_serializer: BaseDocSerializer,
#         doc: DoclingDocument,
#         **kwargs: Any,
#     ) -> SerializationResult:
#         """Emit HTML <table> inside Markdown instead of pipe tables."""
#         params = MarkdownParams(**kwargs)
#         res_parts: list[SerializationResult] = []

#         # optional captions/annotations, identical to Docling's default MD serializer
#         cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
#         if cap_res.text:
#             res_parts.append(cap_res)

#         if item.self_ref in doc_serializer.get_excluded_refs(**kwargs):
#             return create_ser_result(text="\n\n".join([p.text for p in res_parts]), span_source=res_parts)

#         if params.include_annotations:
#             ann_res = doc_serializer.serialize_annotations(item=item, **kwargs)
#             if ann_res.text:
#                 res_parts.append(ann_res)

#         # Build rows from item.data.grid (this is the current Docling API)
#         # Convert RichTableCell via serializer to preserve inline formatting.
#         def cell_text(c):
#             if isinstance(c, RichTableCell):
#                 return doc_serializer.serialize(item=c.ref.resolve(doc=doc), **kwargs).text.replace("\n", " ")
#             return (c.text or "").replace("\n", " ")

#         grid = item.data.grid if item.data and item.data.grid else []

#         # Split header (first row) vs body (remaining rows). If no rows, nothing to do.
#         thead_rows = []
#         tbody_rows = []
#         if grid:
#             thead_rows.append([cell_text(c) for c in grid[0]])
#             for r in grid[1:]:
#                 tbody_rows.append([cell_text(c) for c in r])

#         # Emit HTML table
#         parts = ["<table>"]
#         if thead_rows:
#             parts.append("<thead>")
#             for r in thead_rows:
#                 parts.append("<tr>" + "".join(f"<th>{html.escape(t)}</th>" for t in r) + "</tr>")
#             parts.append("</thead>")
#         parts.append("<tbody>")
#         for r in tbody_rows:
#             parts.append("<tr>" + "".join(f"<td>{html.escape(t)}</td>" for t in r) + "</tr>")
#         parts.append("</tbody></table>")

#         res_parts.append(create_ser_result(text="".join(parts), span_source=item))

#         return create_ser_result(text="\n\n".join([p.text for p in res_parts if p.text]), span_source=res_parts)

# # ---- use it instead of doc.export_to_markdown() ----
# # after you build your `doc: DoclingDocument`
# md_serializer = MarkdownDocSerializer(
#     doc=doc,
#     table_serializer=HtmlTableInMarkdownSerializer(),  # <- key change
#     params=MarkdownParams(
#         image_placeholder="<!-- image -->",
#         # You can adjust params here if needed
#     ),
# )
# markdown_with_html_tables = md_serializer.serialize().text
# print(markdown_with_html_tables)

# # Optional: save to file
# from pathlib import Path
# out_md = Path("out/with_html_tables.md")
# out_md.parent.mkdir(parents=True, exist_ok=True)
# out_md.write_text(markdown_with_html_tables, encoding="utf-8")


# print(f"Markdown:\n{doc.export_to_markdown()}\n")

# ## export as any format.
# # Path("out/").mkdir(parents=True, exist_ok=True)
# # HTML:
# output_path_html = Path("out/") / "example.html"
# # create output directory if not exists
# output_path_html.parent.mkdir(parents=True, exist_ok=True)
# doc.save_as_html(output_path_html)
# # Markdown:
# # output_path_md = Path("out/") / "example.md"
# # doc.save_as_markdown(output_path_md)
