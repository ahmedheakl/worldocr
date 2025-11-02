from pathlib import Path
from vllm import LLM
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
from mineru_vl_utils import MinerUClient
from mineru_vl_utils import MinerULogitsProcessor 


parser = ArgumentParser()
parser.add_argument("--input_dir", type=str, default="../data/omnidocbench_output_mega/images")
parser.add_argument("--output_dir", type=str, default="../data/predictions_mega/mineru25")
parser.add_argument("--model_path", type=str, default="opendatalab/MinerU2.5-2509-1.2B")
args = parser.parse_args()

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
output_dir = Path(args.output_dir)
input_dir = Path(args.input_dir)
output_dir.mkdir(parents=True, exist_ok=True)
valid_exts = {".png", ".jpg", ".jpeg"}
img_paths = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in valid_exts])
if not img_paths: raise SystemExit(f"No images found in {input_dir} with extensions {sorted(valid_exts)}.")

def blocks_to_markdown(extracted_blocks):
    markdown_lines = []
    for block in extracted_blocks:
        block_type = block.get('type', '')
        content = block.get('content', '')
        if not content: continue
        elif block_type == 'title':
            markdown_lines.append(f"# {content}")
        elif block_type == 'code':
            markdown_lines.append("```")
            markdown_lines.append(content)
            markdown_lines.append("```")
        elif block_type == 'algorithm':
            markdown_lines.append("```")
            markdown_lines.append(content)
            markdown_lines.append("```")
        elif block_type == 'image':
            markdown_lines.append(f"![Image](image_{hash(str(block.get('bbox', '')))})")
        elif block_type == 'equation':
            markdown_lines.append(f"${content}$")
        elif block_type == 'equation_block':
            markdown_lines.append("$$")
            markdown_lines.append(content)
            markdown_lines.append("$$")
        elif block_type == 'aside_text':
            markdown_lines.append(f"> {content}")
        else:
            markdown_lines.append(content)
        markdown_lines.append("")
    return "\n".join(markdown_lines)

llm = LLM(
    model=args.model_path,
    logits_processors=[MinerULogitsProcessor] 
)
client = MinerUClient(
    backend="vllm-engine",
    vllm_llm=llm
)

batch_size = 64
for i in range(0, len(img_paths), batch_size):
    in_img_paths = img_paths[i : i + batch_size]
    batched_inputs = []
    output_stems = []  
    for img_path in in_img_paths:
        with Image.open(img_path) as im:
            image = im.convert("RGB")
        batched_inputs.append(image)
        rel = img_path.relative_to(input_dir)
        stem = rel.as_posix().rsplit(".", 1)[0].replace("/", "__")
        output_stems.append(stem)

    outputs = client.batch_two_step_extract(batched_inputs)
    for stem, output in tqdm(zip(output_stems, outputs), desc="Saving outputs", total=len(in_img_paths)):
        markdown = blocks_to_markdown(output)
        md_path = output_dir / f"{stem}.md"
        with md_path.open("w", encoding="utf-8") as f:
            f.write(markdown)

    print(f"Processed {len(in_img_paths)} images.")
    print(f"Markdown saved to: {output_dir.resolve()}")
