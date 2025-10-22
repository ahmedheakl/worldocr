from typing import Any
import html

from docling_core.transforms.serializer.base import BaseDocSerializer, SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownParams, MarkdownTableSerializer
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
        """Emit HTML <table> inside Markdown instead of pipe tables."""
        params = MarkdownParams(**kwargs)
        res_parts: list[SerializationResult] = []

        # optional captions/annotations, identical to Docling's default MD serializer
        cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
        if cap_res.text:
            res_parts.append(cap_res)

        if item.self_ref in doc_serializer.get_excluded_refs(**kwargs):
            return create_ser_result(text="\n\n".join([p.text for p in res_parts]), span_source=res_parts)

        if params.include_annotations:
            ann_res = doc_serializer.serialize_annotations(item=item, **kwargs)
            if ann_res.text:
                res_parts.append(ann_res)

        # Build rows from item.data.grid (this is the current Docling API)
        # Convert RichTableCell via serializer to preserve inline formatting.
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

# ---- use it instead of doc.export_to_markdown() ----
# after you build your `doc: DoclingDocument`
