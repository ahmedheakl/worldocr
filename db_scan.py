import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from sklearn.cluster import DBSCAN
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patches as mpatches


class DocumentElement:
    def __init__(self, bbox: Dict, text: str, element_type: str, element_id: str):
        self.bbox = bbox
        self.text = text
        self.element_type = element_type
        self.element_id = element_id
        self.center_x = (bbox['l'] + bbox['r']) / 2
        self.center_y = (bbox['t'] + bbox['b']) / 2
        self.column = None
        self.reading_order = None


class ColumnReadingOrderDetector:
    def __init__(self, eps_ratio: float = 0.15, min_samples: int = 1):
        self.eps_ratio = eps_ratio
        self.min_samples = min_samples
        
    def load_doctags(self, json_path: str) -> Tuple[List[DocumentElement], Dict]:
        data = json.load(open(json_path, 'r', encoding='utf-8'))
        elements = []
        
        for elem_type, key in [('text', 'texts'), ('picture', 'pictures'), ('table', 'tables')]:
            for idx, elem in enumerate(data.get(key, [])):
                if elem.get('prov') and elem['prov']:
                    bbox = elem['prov'][0]['bbox']
                    text = elem.get('text', '').strip() if elem_type == 'text' else f'[{elem_type.upper()}]'
                    if text:
                        elements.append(DocumentElement(bbox, text, elem_type, f"{elem_type}_{idx}"))
        
        return elements, list(data.get('pages', {}).values())[0] if data.get('pages') else None
    
    def cluster_columns(self, elements: List[DocumentElement], page_width: float) -> List[DocumentElement]:
        if not elements:
            return elements
        
        X = np.array([[e.center_x] for e in elements])
        labels = DBSCAN(eps=self.eps_ratio * page_width, min_samples=self.min_samples).fit(X).labels_
        col_map = {old: new for new, old in enumerate(sorted(set(labels)))}
        
        for elem, label in zip(elements, labels):
            elem.column = col_map[label]
        
        return elements
    
    def assign_reading_order(self, elements: List[DocumentElement]) -> List[DocumentElement]:
        cols = {}
        for e in elements:
            cols.setdefault(e.column, []).append(e)
        
        order = 0
        for col in sorted(cols.keys()):
            for e in sorted(cols[col], key=lambda x: x.center_y):
                e.reading_order = order
                order += 1
        
        return elements
    
    def visualize_reading_order(self, elements: List[DocumentElement], image_path: str, output_path: str, show_text: bool = True):
        img = cv2.imread(image_path)
        if img is None:
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 20))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        sorted_elems = sorted(elements, key=lambda e: e.reading_order)
        
        for e in sorted_elems:
            color = colors[e.column % len(colors)]
            w, h = e.bbox['r'] - e.bbox['l'], e.bbox['b'] - e.bbox['t']
            
            ax.add_patch(FancyBboxPatch((e.bbox['l'], e.bbox['t']), w, h, linewidth=2, 
                                        edgecolor=color, facecolor='none', boxstyle="round,pad=3", alpha=0.8))
            ax.text(e.bbox['l'] + 10, e.bbox['t'] + 20, str(e.reading_order + 1), fontsize=14, 
                   fontweight='bold', color='white', bbox=dict(facecolor=color, alpha=0.9, boxstyle='circle,pad=0.3'))
            
            if show_text and e.element_type == 'text':
                preview = e.text[:30] + '...' if len(e.text) > 30 else e.text
                ax.text(e.bbox['l'] + w + 10, e.bbox['t'] + h / 2, preview, fontsize=8, 
                       color=color, bbox=dict(facecolor='white', alpha=0.7, pad=2))
        
        for i in range(len(sorted_elems) - 1):
            e1, e2 = sorted_elems[i], sorted_elems[i + 1]
            ax.add_patch(FancyArrowPatch((e1.center_x, e1.center_y), (e2.center_x, e2.center_y),
                                        arrowstyle='->,head_width=0.8,head_length=1.2', 
                                        color='red', linewidth=2.5, alpha=0.7, linestyle='--', mutation_scale=30))
        
        ax.legend(handles=[mpatches.Patch(color=colors[i % len(colors)], label=f'Column {i + 1}')
                          for i in range(len(set(e.column for e in elements)))], loc='upper right', fontsize=12)
        ax.set_title('Reading Order Visualization\n(Numbers show reading sequence, arrows show flow)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def export_reading_order(self, elements: List[DocumentElement], output_path: str):
        sorted_elems = sorted(elements, key=lambda e: e.reading_order)
        data = {
            'total_elements': len(elements),
            'num_columns': len(set(e.column for e in elements)),
            'reading_order': [{
                'order': e.reading_order + 1, 'column': e.column + 1, 'element_type': e.element_type,
                'element_id': e.element_id, 'text': e.text[:100] + '...' if len(e.text) > 100 else e.text,
                'bbox': e.bbox, 'center': {'x': e.center_x, 'y': e.center_y}
            } for e in sorted_elems]
        }
        json.dump(data, open(output_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    
    def process_document(self, doctags_path: str, image_path: str, output_dir: str, visualize: bool = True) -> List[DocumentElement]:
        elements, page_info = self.load_doctags(doctags_path)
        if not elements:
            return []
        
        page_width = page_info['size']['width'] if page_info else 2000
        elements = self.assign_reading_order(self.cluster_columns(elements, page_width))
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(doctags_path).stem
        
        self.export_reading_order(elements, os.path.join(output_dir, f"{base_name}_reading_order.json"))
        if visualize and os.path.exists(image_path):
            self.visualize_reading_order(elements, image_path, 
                                        os.path.join(output_dir, f"{base_name}_visualization.jpg"), False)
        
        return elements


def main():
    base_dir = Path("sohail_db_v1")
    detector = ColumnReadingOrderDetector(eps_ratio=0.15, min_samples=1)
    doctags_files = [f for f in (base_dir / "doctags").glob("*.json") if not f.name.endswith('.jsonZone.Identifier')]
    
    if not doctags_files:
        print(f"No doctagss files..")
        return
    print(f"Processing {len(doctags_files)} documents..\n")
    
    results = []
    for doc_path in sorted(doctags_files):
        base_name = doc_path.stem
        img_path = base_dir / "images" / f"{base_name}.jpg" 
        elements = detector.process_document(str(doc_path), str(img_path), 
                                            str(base_dir / "reading_order_results"), True)
        results.append({'document': base_name, 'elements': len(elements), 
                       'columns': len(set(e.column for e in elements)) if elements else 0})
    
    print("...PROCESSING COMPLETE...")
    print(f"\nProcessed {len(results)} document(s)")

if __name__ == "__main__":
    main()
