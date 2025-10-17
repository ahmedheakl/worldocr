import os, re, html, json, hashlib, ast
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm
from PIL import Image
from io import BytesIO

# =========================
# Config
# =========================
ROOT = "data"
PARQUET_PATH = f"{ROOT}/benchmark_100_per_lang_v2.parquet"
DOCTAG_COL   = "doctag_html"
IMAGE_COL    = "image"
OUT_DIR      = f"{ROOT}/docling_json_out"
OUT_DIR_IMAGE = f"{ROOT}/docling_images_out"
OUT_DIR_DOCTAG = f"{ROOT}/docling_doctag_out"
ADD_JSON_COLUMN = False


# =========================
# Utilities
# =========================
def _binary_hash_u64(data: bytes) -> int:
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest[:8], "big", signed=False)

def _infer_mimetype_from_path(path: Optional[str]) -> str:
    if not path:
        return "application/octet-stream"
    ext = Path(path).suffix.lower()
    return {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".tif": "image/tiff", ".tiff": "image/tiff",
        ".bmp": "image/bmp", ".gif": "image/gif",
        ".webp": "image/webp"
    }.get(ext, "application/octet-stream")

def _doc_name_from_path(path: Optional[str], idx: int) -> str:
    return Path(path).stem if path else f"row_{idx:06d}"

def _to_dict_possible_str(x):
    if isinstance(x, dict) or x is None:
        return x
    if isinstance(x, str) and x.strip().startswith("{"):
        try:
            return ast.literal_eval(x)
        except Exception:
            return None
    return None

def extract_bbox_from_tag(tag: Tag) -> Optional[Dict[str, Any]]:
    """Extract bbox from <loc_> tags that precede or wrap the element"""
    # Look for <loc_X> patterns in the tag's previous siblings or in parent
    tag_text = html.unescape(str(tag))
    m = re.search(r'<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>', tag_text)
    locs = m.groups() if m else []
    assert len(locs) == 4, f"Found {len(locs)} <loc_> tags in element; expected exactly 4."
    coords = [int(x) for x in locs[-4:]]
    return {
        "page_no": 1,
        "bbox": {
            "l": coords[0],
            "t": coords[1],
            "r": coords[2],
            "b": coords[3],
            "coord_origin": "TOPLEFT"
        },
        "charspan": [0, len(text_of(tag))]
    }

def unescape_and_clean(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""
    s = html.unescape(raw)
    # remove outer <doctag>
    s = re.sub(r"</?doctag[^>]*>", "", s, flags=re.I)
    # DO NOT remove <loc_> tags here - we need them for bbox extraction
    # normalize whitespace a bit (preserve single spaces)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    return s.strip()

def is_leaf_table(t: Tag) -> bool:
    return t.name == "table" and t.find("tr") and not t.find("table")

def text_of(tag: Tag) -> str:
    # Remove <loc_> tags from text content
    text = tag.get_text(" ", strip=True)
    text = re.sub(r'<loc_\d+>', '', text)
    return re.sub(r"\s+", " ", text).strip()

def mark_processed(tag: Tag):
    tag.attrs["_processed"] = "1"
    for d in tag.descendants:
        if isinstance(d, Tag):
            d.attrs["_processed"] = "1"

def already_processed(tag: Tag) -> bool:
    return isinstance(tag, Tag) and tag.attrs.get("_processed") == "1"

def level_from_header_tag(tag_name: str) -> Optional[int]:
    m = re.match(r"section_header_level_(\d+)$", tag_name)
    return int(m.group(1)) if m else None




# =========================
# Builders (Docling nodes)
# =========================
class Builder:
    def __init__(self):
        self.texts: List[Dict[str, Any]] = []
        self.tables: List[Dict[str, Any]] = []
        self.pictures: List[Dict[str, Any]] = []
        self.groups: List[Dict[str, Any]] = []
        self.body_children: List[Dict[str, Any]] = []

    def _text_ref(self, idx: int) -> Dict[str, Any]:
        return {"$ref": f"#/texts/{idx}"}

    def _table_ref(self, idx: int) -> Dict[str, Any]:
        return {"$ref": f"#/tables/{idx}"}

    def _picture_ref(self, idx: int) -> Dict[str, Any]:
        return {"$ref": f"#/pictures/{idx}"}

    # ---- TEXTS ----
    def add_text(self, label: str, content: str, parent_ref: Dict[str, Any], level: Optional[int] = None, bbox: Optional[Dict[str, Any]] = None) -> int:
        idx = len(self.texts)
        prov = [bbox] if bbox else []
        node = {
            "self_ref": f"#/texts/{idx}",
            "parent": parent_ref,
            "children": [],
            "content_layer": "body",
            "label": label,
            "prov": prov,
            "orig": content,
            "text": content
        }
        if label == "section_header" and level is not None:
            node["level"] = level
        self.texts.append(node)
        # Most texts should appear in reading order; captions are linked to picture/table, not the body.
        if label not in {"caption"}:
            self.body_children.append(self._text_ref(idx))
        return idx

    # ---- TABLES ----
    def add_table_from_grid(self, grid, bbox: Optional[Dict[str, Any]] = None) -> int:
        # grid rows may contain:
        #   - "text"                              → origin 1x1
        #   - (text, rs, cs)                      → origin with spans
        #   - ("", 0, 0)                          → covered cell placeholder
        num_rows = len(grid)
        num_cols = max((len(r) for r in grid), default=0)

        def make_cell(r, c, text, rs=1, cs=1):
            return {
                "row_span": rs, "col_span": cs,
                "start_row_offset_idx": r, "end_row_offset_idx": r + rs,
                "start_col_offset_idx": c, "end_col_offset_idx": c + cs,
                "text": text,
                "column_header": False, "row_header": False,
                "row_section": False, "fillable": False,
            }

        table_cells = []
        grid_cells  = []

        for r in range(num_rows):
            row_cells = []
            for c in range(num_cols):
                item = grid[r][c] if c < len(grid[r]) else ""
                if isinstance(item, tuple):
                    text, rs, cs = item
                    if rs == 0 and cs == 0:
                        # covered cell: render empty 1x1 in the grid; do NOT add to table_cells
                        cell = make_cell(r, c, "", 1, 1)
                    else:
                        cell = make_cell(r, c, text, rs, cs)
                        table_cells.append(cell)  # origin only
                else:
                    # legacy string → origin 1x1
                    cell = make_cell(r, c, str(item), 1, 1)
                    table_cells.append(cell)
                row_cells.append(cell)
            grid_cells.append(row_cells)

        idx = len(self.tables)
        node = {
            "self_ref": f"#/tables/{idx}",
            "parent": {"$ref": "#/body"},
            "children": [],
            "content_layer": "body",
            "label": "table",
            "prov": [bbox] if bbox else [],
            "captions": [],
            "references": [],
            "footnotes": [],
            "data": {
                "table_cells": table_cells,
                "num_rows": num_rows,
                "num_cols": num_cols,
                "grid": grid_cells,
            },
            "annotations": [],
        }
        self.tables.append(node)
        self.body_children.append(self._table_ref(idx))
        return idx


    def link_table_caption(self, table_idx: int, text_idx: int):
        self.tables[table_idx]["captions"].append(self._text_ref(text_idx))
        # Make caption a logical child of the table too:
        self.tables[table_idx]["children"].append(self._text_ref(text_idx))
        # Ensure caption's parent points to the table:
        self.texts[text_idx]["parent"] = self._table_ref(table_idx)

    # ---- PICTURES (FIGURES) ----
    def add_picture(self, bbox: Optional[Dict[str, Any]] = None) -> int:
        idx = len(self.pictures)
        prov = [bbox] if bbox else []
        node = {
            "self_ref": f"#/pictures/{idx}",
            "parent": {"$ref": "#/body"},
            "children": [],
            "content_layer": "body",
            "label": "picture",
            "prov": prov,
            "captions": [],
            "references": [],
            "footnotes": [],
            "annotations": []
        }
        self.pictures.append(node)
        self.body_children.append(self._picture_ref(idx))
        return idx

    def link_picture_caption(self, picture_idx: int, text_idx: int):
        self.pictures[picture_idx]["captions"].append(self._text_ref(text_idx))
        self.pictures[picture_idx]["children"].append(self._text_ref(text_idx))
        self.texts[text_idx]["parent"] = self._picture_ref(picture_idx)

    # ---- GROUPS (LISTS) ----
    def add_group(self, name: str, child_text_indices: List[int]) -> int:
        idx = len(self.groups)
        node = {
            "self_ref": f"#/groups/{idx}",
            "parent": {"$ref": "#/body"},
            "children": [self._text_ref(i) for i in child_text_indices],
            "content_layer": "furniture",
            "name": "group",
            "label": name
        }
        self.groups.append(node)
        self.body_children.append({"$ref": f"#/groups/{idx}"})
        return idx


# =========================
# Extraction
# =========================
def extract_tables(soup: BeautifulSoup, b: Builder) -> List[int]:
    for t in soup.find_all("table"):
        if already_processed(t):
            continue
        if not is_leaf_table(t):
            bbox = extract_bbox_from_tag(t)
            inner_table = t.find("table")
            if inner_table and is_leaf_table(inner_table):
                t = inner_table
        else:
            bbox = extract_bbox_from_tag(t)

        grid = []
        open_rowspans: List[int] = []  # per column: remaining rows covered

        for tr in t.find_all("tr", recursive=False):
            row: List[Any] = []
            col = 0

            # advance into placeholders for any leading covered columns
            while col < len(open_rowspans) and open_rowspans[col] > 0:
                row.append(("", 0, 0))
                col += 1

            for td in tr.find_all(["td", "th"], recursive=False):
                text = text_of(td)
                cs = int(td.attrs.get("colspan", "1") or "1")
                rs = int(td.attrs.get("rowspan", "1") or "1")

                # move past covered columns
                while col < len(open_rowspans) and open_rowspans[col] > 0:
                    row.append(("", 0, 0)); col += 1

                need = col + cs
                if len(open_rowspans) < need:
                    open_rowspans.extend([0] * (need - len(open_rowspans)))

                # place origin
                row.append((text, rs, cs))
                # same-row placeholders for colspan
                for _ in range(cs - 1):
                    row.append(("", 0, 0))

                # mark future coverage for rowspan
                if rs > 1:
                    for k in range(col, col + cs):
                        open_rowspans[k] = max(open_rowspans[k], rs - 1)

                col += cs

            # trailing covered columns in this row → placeholders
            for k in range(col, len(open_rowspans)):
                if open_rowspans[k] > 0:
                    row.append(("", 0, 0))

            grid.append(row)
            # decrement coverage for next row
            open_rowspans = [max(0, x - 1) for x in open_rowspans]

        # normalize row widths (pad with empty 1x1 cells)
        max_cols = max((len(r) for r in grid), default=0)
        for r in grid:
            if len(r) < max_cols:
                r.extend([""] * (max_cols - len(r)))

        b.add_table_from_grid(grid, bbox)
        mark_processed(t)


def extract_lists(soup: BeautifulSoup, b: Builder):
    # <list> usually contains <text> items → make a group that references those texts
    for lst in soup.find_all("list"):
        if already_processed(lst):
            continue
        item_text_ids: List[int] = []
        # Prefer direct <text> children; otherwise, split non-empty lines
        texts = lst.find_all("text", recursive=True)
        if texts:
            for t in texts:
                if already_processed(t):
                    continue
                content = text_of(t)
                if content:
                    bbox = extract_bbox_from_tag(t)
                    tidx = b.add_text("list_item", content, {"$ref": "#/body"}, bbox=bbox)
                    item_text_ids.append(tidx)
                mark_processed(t)
            
        else:
            raw = text_of(lst)
            bbox = extract_bbox_from_tag(lst)
            for line in [x.strip() for x in raw.split("\n") if x.strip()]:
                tidx = b.add_text("list_item", line, {"$ref": "#/body"}, bbox=bbox)
                item_text_ids.append(tidx)

        # CHANGEME: Currently disabled because it creates too many groups
        b.add_group("list", item_text_ids)
        mark_processed(lst)

def extract_figures_and_captions(soup: BeautifulSoup, b: Builder):
    # Process <figure> blocks; attach nested <figure_caption> as caption
    for fig in soup.find_all("figure"):
        if already_processed(fig):
            continue
        bbox = extract_bbox_from_tag(fig)
        pic_idx = b.add_picture(bbox)
        element_range = 1 # how many elements previous or following to search for a caption
        prev_cap = fig.find_previous("figure_caption")
        following_cap = fig.find_next("figure_caption")
        for _ in range(element_range):
            if prev_cap and not already_processed(prev_cap):
                cap_text = text_of(prev_cap)
                cap_bbox = extract_bbox_from_tag(prev_cap)
                cap_idx = b.add_text("caption", cap_text, b._picture_ref(pic_idx), bbox=cap_bbox)
                b.link_picture_caption(pic_idx, cap_idx)
                mark_processed(prev_cap)
                break
            if following_cap and not already_processed(following_cap):
                cap_text = text_of(following_cap)
                cap_bbox = extract_bbox_from_tag(following_cap)
                cap_idx = b.add_text("caption", cap_text, b._picture_ref(pic_idx), bbox=cap_bbox)
                b.link_picture_caption(pic_idx, cap_idx)
                mark_processed(following_cap)
                break
            if prev_cap:
                prev_cap = prev_cap.find_previous("figure_caption")
            if following_cap:
                following_cap = following_cap.find_next("figure_caption")
        mark_processed(fig)

    # Standalone <figure_caption> (rare): attach to the last picture if exists
    for cap in soup.find_all("figure_caption"):
        if already_processed(cap):
            continue
        elements_range = 2
        cap_text = text_of(cap)
        prev_figure = cap.find_previous("figure")
        following_figure = cap.find_next("figure")
        last_pic = -1
        for _ in range(elements_range):
            if prev_figure and not already_processed(prev_figure):
                cur_bbox = extract_bbox_from_tag(prev_figure)['bbox']
                last_pic = None
                for pi, pnode in enumerate(b.pictures):
                    if pnode["prov"] and pnode["prov"][0]["bbox"] == cur_bbox:
                        last_pic = pi
                        break
                if last_pic is not None:
                    break
            if following_figure and not already_processed(following_figure):
                cur_bbox = extract_bbox_from_tag(following_figure)['bbox']
                last_pic = None
                for pi, pnode in enumerate(b.pictures):
                    if pnode["prov"] and pnode["prov"][0]["bbox"] == cur_bbox:
                        last_pic = pi
                        break
                if last_pic is not None:
                    break
            if prev_figure:
                prev_figure = prev_figure.find_previous("figure")
            if following_figure:
                following_figure = following_figure.find_next("figure")
        cap_bbox = extract_bbox_from_tag(cap)
        if last_pic >= 0:
            cap_idx = b.add_text("caption", cap_text, b._picture_ref(last_pic), bbox=cap_bbox)
            b.link_picture_caption(last_pic, cap_idx)
        else:
            b.add_text("caption", cap_text, {"$ref": "#/body"}, bbox=cap_bbox)
        mark_processed(cap)

def extract_table_captions_in_order(soup: BeautifulSoup, b: Builder):
    for cap in soup.find_all("table_caption"):
        if already_processed(cap):
            continue
        cap_text = text_of(cap)
        if not cap_text:
            mark_processed(cap)
            continue
        cap_bbox = extract_bbox_from_tag(cap)
        elements_range = 2 # how many elements previous or following to search for a table
        target_idx = None
        prev_tbl = cap.find_previous("table")
        following_tbl = cap.find_next("table")
        for _ in range(elements_range):
            if prev_tbl and not is_leaf_table(prev_tbl):
                cur_bbox = extract_bbox_from_tag(prev_tbl)['bbox']
                for ti, tnode in enumerate(b.tables):
                    if tnode["prov"] and tnode["prov"][0]["bbox"] == cur_bbox:
                        target_idx = ti
                        break
                if target_idx is not None:
                    break
            if following_tbl and not is_leaf_table(following_tbl):
                cur_bbox = extract_bbox_from_tag(following_tbl)['bbox']
                for ti, tnode in enumerate(b.tables):
                    if tnode["prov"] and tnode["prov"][0]["bbox"] == cur_bbox:
                        target_idx = ti
                        break
                if target_idx is not None:
                    break
            if prev_tbl:
                prev_tbl = prev_tbl.find_previous("table")
            if following_tbl:
                following_tbl = following_tbl.find_next("table")
        if target_idx is not None:
            cap_idx = b.add_text("caption", cap_text, b._table_ref(target_idx), bbox=cap_bbox)
            b.link_table_caption(target_idx, cap_idx)
        else:
            b.add_text("caption", cap_text, {"$ref": "#/body"}, bbox=cap_bbox)

        mark_processed(cap)


def extract_headers_texts_misc(soup: BeautifulSoup, b: Builder):
    # Section headers (with level)
    for node in soup.find_all(True, recursive=True):
        if already_processed(node):
            continue
        name = node.name or ""
        lvl = level_from_header_tag(name)
        if lvl is None: continue
        content = text_of(node)
        if content:
            bbox = extract_bbox_from_tag(node)
            b.add_text("section_header", content, {"$ref": "#/body"}, level=lvl, bbox=bbox)
        mark_processed(node)

    # Title → treat as section_header level 0
    for node in soup.find_all("title"):
        if already_processed(node): continue
        content = text_of(node)
        if content:
            bbox = extract_bbox_from_tag(node)
            b.add_text("title", content, {"$ref": "#/body"}, bbox=bbox)
        mark_processed(node)

    # Header/Footer
    for node in soup.find_all("header"):
        if already_processed(node): continue
        content = text_of(node)
        if content:
            bbox = extract_bbox_from_tag(node)
            b.add_text("page_header", content, {"$ref": "#/body"}, bbox=bbox)
        mark_processed(node)

    for node in soup.find_all("footer"):
        if already_processed(node): continue
        content = text_of(node)
        if content:
            bbox = extract_bbox_from_tag(node)
            b.add_text("page_footer", content, {"$ref": "#/body"}, bbox=bbox)
        mark_processed(node)

    # TOC → convert to a group of lines (like a list)
    for toc in soup.find_all("toc"):
        if already_processed(toc): continue
        lines = [ln.strip() for ln in text_of(toc).split("\n") if ln.strip()]
        bbox = extract_bbox_from_tag(toc)
        tids = [b.add_text("text", ln, {"$ref": "#/body"}, bbox=bbox) for ln in lines]
        b.add_group("list", tids)
        mark_processed(toc)

    # Quote / Equation / Generic text
    for node in soup.find_all(["quote", "equation", "text"]):
        if already_processed(node): continue
        label = "formula" if node.name == "equation" else "text"
        content = text_of(node)
        bbox = extract_bbox_from_tag(node)
        b.add_text(label, content, {"$ref": "#/body"}, bbox=bbox)
        mark_processed(node)
        
        
# =========================
# Reading order
# =========================
def _compute_body_children_in_dom_order(soup: BeautifulSoup, b: Builder) -> List[Dict[str, Any]]:
    """
    Rebuild body.children in the original DOM reading order.
    We match built nodes to source tags via their bbox in prov[0].bbox.
    Captions are naturally excluded because their parent is the figure/table, not #/body.
    """
    body_children: List[Dict[str, Any]] = []
    used_refs = set()  # to avoid duplicates

    def _ref_key(ref: Dict[str, Any]) -> str:
        return ref.get("$ref", "")

    def _push_ref(ref: Dict[str, Any]):
        k = _ref_key(ref)
        if k and k not in used_refs:
            used_refs.add(k)
            body_children.append(ref)

    def _bbox_of_tag(tag: Tag) -> Optional[Dict[str, Any]]:
        try:
            return extract_bbox_from_tag(tag)["bbox"]
        except Exception:
            return None

    def _find_picture_by_bbox(bbox) -> Optional[int]:
        if bbox is None: return None
        for i, node in enumerate(b.pictures):
            if node.get("prov") and node["prov"][0].get("bbox") == bbox:
                return i
        return None

    def _find_table_by_bbox(bbox) -> Optional[int]:
        if bbox is None: return None
        for i, node in enumerate(b.tables):
            if node.get("prov") and node["prov"][0].get("bbox") == bbox:
                return i
        return None

    def _find_texts_by_bbox_and_labels(bbox, labels: Optional[set] = None) -> List[int]:
        """
        Return text indices whose parent is #/body, whose label matches (if provided),
        and whose first prov bbox equals bbox, preserving their creation order.
        """
        out = []
        for i, node in enumerate(b.texts):
            if node.get("parent", {}).get("$ref") != "#/body":
                continue  # skip captions etc.
            if labels and node.get("label") not in labels:
                continue
            prov = node.get("prov") or []
            if prov and prov[0].get("bbox") == bbox:
                out.append(i)
        return out

    # Which DOM nodes should drive body order?
    def _is_interesting(node: Tag) -> bool:
        if not isinstance(node, Tag): return False
        if node.name in {"table", "figure", "list", "toc", "text", "quote", "equation", "title", "header", "footer"}:
            return True
        # dynamic section headers: section_header_level_N
        return level_from_header_tag(node.name or "") is not None

    # Walk the DOM in source order and push corresponding refs
    for node in soup.find_all(True, recursive=True):
        if not _is_interesting(node):
            continue
        bbox = _bbox_of_tag(node)

        if node.name == "table":
            ti = _find_table_by_bbox(bbox)
            if ti is not None:
                _push_ref(b._table_ref(ti))

        elif node.name == "figure":
            pi = _find_picture_by_bbox(bbox)
            if pi is not None:
                _push_ref(b._picture_ref(pi))

        elif node.name == "list":
            for tid in _find_texts_by_bbox_and_labels(bbox, labels={"text", "list_item"}):
                _push_ref(b._text_ref(tid))

        elif node.name == "toc":
            # TOC also materialized as multiple plain text nodes
            for tid in _find_texts_by_bbox_and_labels(bbox, labels={"text"}):
                _push_ref(b._text_ref(tid))

        elif node.name in {"title"} or level_from_header_tag(node.name or "") is not None:
            for tid in _find_texts_by_bbox_and_labels(bbox, labels={"section_header", "title"}):
                _push_ref(b._text_ref(tid))

        elif node.name in {"header"}:
            for tid in _find_texts_by_bbox_and_labels(bbox, labels={"page_header"}):
                _push_ref(b._text_ref(tid))

        elif node.name in {"footer"}:
            for tid in _find_texts_by_bbox_and_labels(bbox, labels={"page_footer"}):
                _push_ref(b._text_ref(tid))

        elif node.name in {"quote"}:
            for tid in _find_texts_by_bbox_and_labels(bbox, labels={"quote"}):
                _push_ref(b._text_ref(tid))

        elif node.name in {"equation"}:
            for tid in _find_texts_by_bbox_and_labels(bbox, labels={"equation"}):
                _push_ref(b._text_ref(tid))

        elif node.name in {"text"}:
            for tid in _find_texts_by_bbox_and_labels(bbox, labels={"text"}):
                _push_ref(b._text_ref(tid))

        # NOTE: figure_caption / table_caption are intentionally NOT pushed;
        # they live under their figure/table parent and should not appear in body.children.

    return body_children


def parse_doctag_to_docling(doc_html: str, img_meta_in: Optional[Dict[str, Any]], row_idx: int) -> Dict[str, Any]:
    s = unescape_and_clean(doc_html)
    soup = BeautifulSoup(s, "lxml")  # robust parser

    b = Builder()
    # FIXME: this style destroys the original reading order, we need to do better
    # e.g. <figure> after <text> will appear before it in the body
    # 1) Lists → groups of text items
    extract_lists(soup, b)

    # 2) Tables
    extract_tables(soup, b)

    # 3) Figures and captions
    extract_figures_and_captions(soup, b)

    # 4) Table captions (attach to nearest previous table)
    extract_table_captions_in_order(soup, b)

    # 5) Headers, TOC, quotes, equations, remaining text
    extract_headers_texts_misc(soup, b)
    new_soup = BeautifulSoup(s, "lxml")  # re-parse to get updated _processed attrs
    b.body_children = _compute_body_children_in_dom_order(new_soup, b)

    # ---- Origin from image column ----
    img_meta = _to_dict_possible_str(img_meta_in)
    filename = img_meta.get("path") if img_meta else None
    mimetype = _infer_mimetype_from_path(filename)
    binary_hash = None
    if img_meta and isinstance(img_meta.get("bytes"), (bytes, bytearray)):
        binary_hash = _binary_hash_u64(bytes(img_meta["bytes"]))
    w, h = Image.open(BytesIO(img_meta["bytes"])).size
    name = _doc_name_from_path(filename, row_idx)
    doc = {
        "schema_name": "DoclingDocument",
        "version": "1.7.0",
        "name": name,
        "origin": {
            "mimetype": mimetype,
            "binary_hash": binary_hash,
            "filename": os.path.basename(filename) if filename else None,
        },
        "furniture": {
            "self_ref": "#/furniture",
            "children": [],
            "content_layer": "furniture",
            "name": "_root_",
            "label": "unspecified",
        },
        "body": {
            "self_ref": "#/body",
            "children": b.body_children,
            "content_layer": "body",
            "name": "_root_",
            "label": "unspecified",
        },
        "groups": b.groups,
        "texts": b.texts,
        "pictures": b.pictures,
        "tables": b.tables,
        "key_value_items": [],
        "form_items": [],
        "pages": {
            "1": { "size": {"width": w, "height": h}, "page_no": 1 }
        }
    }
    return doc


# =========================
# Runner
# =========================
def main():
    df = pd.read_parquet(PARQUET_PATH)
    if DOCTAG_COL not in df.columns:
        raise KeyError(f"Column '{DOCTAG_COL}' not found.")
    if IMAGE_COL not in df.columns:
        df[IMAGE_COL] = None

    outdir = Path(OUT_DIR); outdir.mkdir(parents=True, exist_ok=True)

    docs = []
    for i, row in tqdm(df.reset_index(drop=True).iterrows(), total=len(df), desc="Processing rows"):
        raw = row.get(DOCTAG_COL, "")
        doc = parse_doctag_to_docling(raw, row[IMAGE_COL], i)
        img_meta = _to_dict_possible_str(row.get(IMAGE_COL, None))
        if img_meta and isinstance(img_meta.get("bytes"), (bytes, bytearray)):
            img_bytes = bytes(img_meta["bytes"])
            img_hash = doc["origin"].get("binary_hash")
            if img_hash is not None:
                img_filename = OUT_DIR_IMAGE + f"/{doc['name']}" + (Path(img_meta.get("path") or "").suffix or ".bin")
                img_path = Path(img_filename)
                img_path.parent.mkdir(parents=True, exist_ok=True)
                with open(img_path, "wb") as imgf:
                    imgf.write(img_bytes)
        # save doctag as well
        if raw.strip(): 
            with open(outdir / f"{doc['name'] or f'row_{i:06d}'}.html", "w", encoding="utf-8") as f:
                f.write(raw)            
        docs.append(doc)
        with open(outdir / f"{doc['name'] or f'row_{i:06d}'}.json", "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

    if ADD_JSON_COLUMN:
        df["docling_json"] = docs
        with open(outdir / "docling_docs.jsonl", "w", encoding="utf-8") as jf:
            for d in docs:
                jf.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(docs)} docs to {outdir.resolve()}")

if __name__ == "__main__":
    main()