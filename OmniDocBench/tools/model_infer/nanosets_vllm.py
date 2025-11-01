# run vllm serve nanonets/Nanonets-OCR-s --gpu-memory-utilization 0.9
import os
import asyncio
from argparse import ArgumentParser
from openai import AsyncOpenAI
import base64
from tqdm import tqdm

async def ocr_page_with_nanonets_s(client, image_path):
    with open(image_path, "rb") as image_file:
        img_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    
    response = await client.chat.completions.create(
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
        max_tokens=12000
    )
    return response.choices[0].message.content

async def process_batch(client, batch, output_folder):
    tasks = []
    for filename, image_path in batch:
        tasks.append(ocr_page_with_nanonets_s(client, image_path))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for (filename, image_path), result in zip(batch, results):
        try:
            if isinstance(result, Exception):
                print(f"Error processing {filename}: {str(result)}")
                continue
            
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.md')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
            
            print(f"Saved result to: {output_path}")
        except Exception as e:
            print(f"Error saving {filename}: {str(e)}")

async def process_folder(input_folder, output_folder, batch_size=8):
    os.makedirs(output_folder, exist_ok=True)
    client = AsyncOpenAI(api_key="123", base_url="http://localhost:8000/v1")
    
    # Collect all image files
    image_files = []
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_path = os.path.join(input_folder, filename)
            image_files.append((filename, image_path))

    for i in tqdm(range(0, len(image_files), batch_size)):
        batch = image_files[i:i + batch_size]
        await process_batch(client, batch, output_folder)

parser = ArgumentParser()
parser.add_argument("--input_dir", type=str, default="../data/omnidocbench_output_en/images")
parser.add_argument("--output_dir", type=str, default="../data/predictions/nanosets")
parser.add_argument("--batch_size", type=int, default=16, help="Number of images to process concurrently")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
asyncio.run(process_folder(args.input_dir, args.output_dir, args.batch_size))