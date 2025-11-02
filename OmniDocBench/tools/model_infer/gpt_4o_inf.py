import random
import tqdm
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import argparse
import os
import base64
from tqdm import tqdm
random.seed(42)

PROMPT = """ You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

    1. Text Processing:
    - Accurately recognize all text content in the PDF image without guessing or inferring.
    - Convert the recognized text into Markdown format.
    - Maintain the original document structure, including headings, paragraphs, lists, etc.

    2. Mathematical Formula Processing:
    - Convert all mathematical formulas to LaTeX format.
    - Enclose inline formulas with \( \). For example: This is an inline formula \( E = mc^2 \)
    - Enclose block formulas with \\[ \\]. For example: \[ \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \]

    3. Table Processing:
    - Convert tables to HTML format.
    - Wrap the entire table with <table> and </table>.

    4. Figure Handling:
    - Ignore figures content in the PDF image. Do not attempt to describe or convert images.

    5. Output Format:
    - Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
    - For complex layouts, try to maintain the original document's structure and format as closely as possible.

    Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.
"""

def get_gpt_response(image_path, model_path="gpt-4o"):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    img_str = base64.b64encode(image_bytes).decode()
    try:
        client = OpenAI()
        completion = client.chat.completions.create(
            model=model_path,
            messages=[
                {"role": "user", "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_str}"
                        }
                    },
                    {"type": "text", "text": PROMPT}
                ]}
            ],
            max_completion_tokens=8192,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"[ERROR] Failed to get response: {e}")
        return ""

def process_image(args):
    image_path, save_root, model_path = args
    file_name = os.path.basename(image_path)
    try:
        response = get_gpt_response(image_path, model_path)
        output_path = os.path.join(save_root, file_name[:-4] + ".md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response)
        return f"success: {file_name}"
    except Exception as e:
        return f"failed {file_name}: {str(e)}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/mnt/hwfile/doc_parse/renzhifei/data/table_mask")
    parser.add_argument("--output_dir", type=str, default="/mnt/hwfile/doc_parse/oylk/model_mds/Omnidocbench_patch_table/GPT-4o")
    parser.add_argument("--model_path", type=str, default="gpt-4o")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    
    image_root = args.input_dir
    save_root = args.output_dir
    num_threads = args.threads
    
    os.makedirs(save_root, exist_ok=True)
    
    image_files = []
    for file in os.listdir(image_root):
        if file.endswith(".jpg") or file.endswith(".png"):
            image_path = os.path.join(image_root, file)
            image_files.append((image_path, save_root, args.model_path))
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(tqdm(executor.map(process_image, image_files), total=len(image_files), desc="Infer"))
    

if __name__ == "__main__":
    main()