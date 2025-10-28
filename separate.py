import json
import os


root1 = "data/omnidocbench_output_med"
outfolder = "spanish_data"
results = "OmniDocBench/result/qwen25vl3b-lorav1_quick_match_text_block_per_page_edit.json"
os.makedirs(outfolder, exist_ok=True)

with open(results, "r", encoding="utf-8") as f:
    results_data = json.load(f)

path = "data/omnidocbench_output_med/cleaned_omnidocbench.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
    
for d in data:
    if d['page_info']['page_attribute']['language'] != "spanish": continue
    image_path = d['page_info']['image_path']
    image_path = os.path.join(root1, image_path)
    md_path = os.path.join("data/predictions_med/qwen25vl3b-lorav1", os.path.basename(image_path).replace(".jpg", ".md"))
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
        
    # replace "```markdown\n{content}\n```" with just {content}
    md_content = md_content.replace("```markdown\n", "").replace("\n```", "")  
    try:
        result = results_data[os.path.basename(image_path)]
    except:
        print(f"Skipping {image_path} as no result found")
        continue
    md_content += f"{result}"
    
    out_image_path = os.path.join(outfolder, os.path.basename(image_path))
    out_md_path = out_image_path.replace(".jpg", ".md")
    os.system(f"cp '{image_path}' '{out_image_path}'")
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    