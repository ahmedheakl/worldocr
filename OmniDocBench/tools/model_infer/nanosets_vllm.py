# run vllm serve nanonets/Nanonets-OCR-s --gpu-memory-utilization 0.9
import os
from tqdm import tqdm
from argparse import ArgumentParser

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in tqdm(os.listdir(input_folder)):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')): continue
        try:
            image_path = os.path.join(input_folder, filename)
            result = ocr_page_with_nanonets_s(image_path)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.md')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
            
            print(f"Saved result to: {output_path}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

from openai import OpenAI
import base64
def ocr_page_with_nanonets_s(image_path):
    client = OpenAI(api_key="123", base_url="http://localhost:8000/v1")
    with open(image_path, "rb") as image_file:
        img_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    response = client.chat.completions.create(
        model="nanonets/Nanonets-OCR-s",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    },
                    {
                        "type": "text",
                        "text": "Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes.",
                    },
                ],
            }
        ],
        temperature=0.0,
        max_tokens=15000
    )
    return response.choices[0].message.content

parser = ArgumentParser()
parser.add_argument("--input_folder", type=str, default="../data/omnidocbench_output_en/images", help="Path to the input folder containing images.")
parser.add_argument("--output_folder", type=str, default="../data/predictions/nanosets", help="Path to the output folder to save markdown files.")
args = parser.parse_args()
os.makedirs(args.output_folder, exist_ok=True)
process_folder(args.input_folder, args.output_folder)