import json
from typing import Dict, Any, List
from vllm import LLM, SamplingParams
from pathlib import Path
import os
from tqdm import tqdm
from bs4 import BeautifulSoup
import re



class DocumentClassifier:
    def __init__(self, model_path: str):
        print(f"Loading model: {model_path}")
        allowed_path = Path("data").resolve()
        self.llm = LLM(
            model=model_path,
            max_model_len=32768,
            max_num_seqs=1,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
            allowed_local_media_path=str(allowed_path)
        )
            
        self.schema = {
            "data_source": [
                "academic_literature",
                "PPT2PDF", 
                "book",
                "colorful_textbook",
                "exam_paper",
                "note",
                "magazine",
                "research_report",
                "newspaper"
            ],
            "layout": [
                "single_column",
                "double_column", 
                "three_column",
                "1andmore_column",
                "other_layout"
            ],
            "watermark": ["yes", "no"],
            "fuzzy_scan": ["yes", "no"],
            "colorful_background": ["yes", "no"]
        }
        
    def create_prompt(self) -> str:
        prompt = """Analyze this document page image and classify it according to the following criteria. Return your answer as a JSON object.

Classification criteria:

1. data_source (PDF type classification):
   - academic_literature: Academic papers and literature
   - PPT2PDF: PowerPoint converted to PDF
   - book: Black and white books and textbooks
   - colorful_textbook: Colorful textbooks with images
   - exam_paper: Exam papers and test sheets
   - note: Handwritten notes
   - magazine: Magazines
   - research_report: Research reports and financial reports
   - newspaper: Newspapers

2. layout (Page layout type):
   - single_column: Single column layout
   - double_column: Double column layout
   - three_column: Three column layout
   - 1andmore_column: One column mixed with multiple columns (common in literature)
   - other_layout: Other layouts

3. watermark: Does the page contain a watermark? (yes/no)

4. fuzzy_scan: Is the page blurry or poorly scanned? (yes/no)

5. colorful_background: Does the content have more than two background colors? (yes/no)

Return ONLY a valid JSON object with these exact keys:
{
    "data_source": "<classification>",
    "layout": "<classification>",
    "watermark": "<yes/no>",
    "fuzzy_scan": "<yes/no>",
    "colorful_background": "<yes/no>"
}"""
        return prompt
    
    def classify_image(self, image_path: str, temperature: float = 0.1) -> Dict[str, Any]:
        prompt = self.create_prompt()
        image_path = Path(image_path).resolve().as_uri()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_path}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=512,
            top_p=0.9,
        )
        outputs = self.llm.chat(
            messages=messages,
            sampling_params=sampling_params,
            use_tqdm=False
        )
        response_text = outputs[0].outputs[0].text.strip()
        try:
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            result = json.loads(response_text)
            self._validate_result(result)
            return result
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {response_text}")
            print(f"Error: {e}")
            return self._get_default_result()
    
    def classify_batch(self, image_paths: List[str], temperature: float = 0.1) -> List[Dict[str, Any]]:
        prompt = self.create_prompt()
        
        conversations = []
        for image_path in image_paths:
            image_path = Path(image_path).resolve().as_uri()
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_path}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            conversations.append(messages)
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=512,
            top_p=0.9,
        )
        
        outputs = self.llm.chat(
            messages=conversations,
            sampling_params=sampling_params,
            # use_tqdm=True
        )
        
        results = []
        for i, output in enumerate(outputs):
            response_text = output.outputs[0].text.strip()
            try:
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                
                result = json.loads(response_text)
                self._validate_result(result)
                results.append(result)
                
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Failed to parse response for image {i} ({image_paths[i]}): {e}")
                results.append(self._get_default_result())
        
        return results
    
    def _validate_result(self, result: Dict[str, Any]) -> None:
        required_keys = ["data_source", "layout", "watermark", "fuzzy_scan", "colorful_background"]
        
        for key in required_keys:
            if key not in result:
                raise ValueError(f"Missing required key: {key}")
            
            if key in ["watermark", "fuzzy_scan", "colorful_background"]:
                if result[key] not in ["yes", "no"]:
                    raise ValueError(f"Invalid value for {key}: {result[key]}")
            elif key == "data_source":
                if result[key] not in self.schema["data_source"]:
                    raise ValueError(f"Invalid data_source: {result[key]}")
            elif key == "layout":
                if result[key] not in self.schema["layout"]:
                    raise ValueError(f"Invalid layout: {result[key]}")
    
    def _get_default_result(self) -> Dict[str, Any]:
        return {
            "data_source": "academic_literature",
            "layout": "single_column",
            "watermark": "no",
            "fuzzy_scan": "no",
            "colorful_background": "no"
        }


def analyze_html_table(html_content: str):
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table')
    if table is None:
        raise ValueError("No table found in HTML.")

    merged = any(cell.has_attr('rowspan') or cell.has_attr('colspan')
                 for cell in table.find_all(['td', 'th']))

    border = "wireless_line"  # default: no borders
    style = table.get('style', '').lower()
    border_attr = table.get('border')

    if border_attr and border_attr != "0":
        border = "full_line"
    elif 'border-collapse' in style or 'border:' in style:
        sides = len(re.findall(r'border-(top|bottom|left|right)\s*:', style))
        if sides == 4:
            border = "full_line"
        elif 2 <= sides <= 3:
            border = "fewer_line"
        elif sides == 1:
            border = "less_line"
        else:
            border = "wireless_line"
    else:
        # Check inlined cell borders
        td_styles = [td.get('style', '') for td in table.find_all(['td', 'th'])]
        bordered_cells = [s for s in td_styles if 'border' in s]
        if bordered_cells:
            border = "less_line"

    text = table.get_text(separator=' ')
    # include_equation = bool(re.search(r'[\=\+\-\*/]', text))

    return {
        'with_span': merged,
        'line': border,
        'include_equation': False
    }



def main():
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    classifier = DocumentClassifier(model_path=model_name)
    root = "data/omnidocbench_output_en"
    annotations_path = os.path.join(root, "omnidocbench.json")
    per_batch = 4
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    for idx in tqdm(range(0, len(annotations), per_batch)):
        batch = []
        for j in range(per_batch):
            if idx + j < len(annotations):
                img_path = os.path.join(root, annotations[idx + j]['page_info']['image_path'])
                batch.append(img_path)
        results = classifier.classify_batch(batch)
        for j in range(len(batch)):
            if idx + j < len(annotations):
                annotations[idx + j]['page_info']['page_attribute']['data_source'] = results[j]['data_source']
                annotations[idx + j]['page_info']['page_attribute']['layout'] = results[j]['layout']
                if results[j]['watermark'] == "yes":
                    annotations[idx + j]['page_info']['page_attribute']['special_issue'].append("watermark")
                if results[j]['fuzzy_scan'] == "yes":
                    annotations[idx + j]['page_info']['page_attribute']['special_issue'].append("fuzzy_scan")
                if results[j]['colorful_background'] == "yes":
                    annotations[idx + j]['page_info']['page_attribute']['special_issue'].append("colorful_background")
                    
    for d in tqdm(annotations):
        for layout_det in d['layout_dets']:
            if layout_det['category_type'] != "table":
                continue
            stats = analyze_html_table(layout_det['html'])
            layout_det['attribute'] = {
            'with_span': stats['with_span'],
            'line': stats['line'],
            'include_equation': stats['include_equation'],
            'language': "table_en",
            'include_photo': False,
            'include_background': True,
            'table_layout': "horizontal",
            }
                
                
    out_path = os.path.join(root, "omnidocbench_enriched.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
            
if __name__ == "__main__":
    main()