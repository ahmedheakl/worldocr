
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from tqdm import tqdm
import json
import argparse
import shutil
from dots_ocr import DotsOCRParser


def download():
    model_dir = "../data/weights/DotsOCR"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="rednote-hilab/dots.ocr", local_dir=model_dir, local_dir_use_symlinks=False, resume_download=True)
    print(f"model downloaded to {model_dir}")

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(
        description="dots.ocr Multilingual Document Layout Parser",
    )
    parser.add_argument(
        "--filepath",
        type=str,
        
    )
    parser.add_argument(
        '--bbox', 
        type=int, 
        nargs=4, 
        metavar=('x1', 'y1', 'x2', 'y2'),
        help='should give this argument if you want to prompt_grounding_ocr'
    )
    parser.add_argument(
        "--ip", type=str, default="localhost",
        help=""
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help=""
    )
    parser.add_argument(
        "--model_name", type=str, default="model",
        help=""
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help=""
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0,
        help=""
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help=""
    )
    parser.add_argument(
        "--max_completion_tokens", type=int, default=16384,
        help=""
    )
    parser.add_argument(
        "--num_thread", type=int, default=128,
        help=""
    )
    parser.add_argument(
        "--min_pixels", type=int, default=None,
        help=""
    )
    parser.add_argument(
        "--max_pixels", type=int, default=None,
        help=""
    )
    parser.add_argument(
        "--eval_result_save_dir", type=str, default="./output_omni/",
    )
    args = parser.parse_args()
    download()
    dots_ocr_parser = DotsOCRParser(
        ip=args.ip,
        port=args.port,
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_completion_tokens=args.max_completion_tokens,
        num_thread=args.num_thread,
        dpi=args.dpi,
        # output_dir=args.output, 
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        use_hf=True,
    )

    with open(args.filepath, 'r') as f:
        list_items = json.load(f)

    results = []
    output_path = "./output_omni.jsonl"
    f_out = open(output_path, 'w')
    root_dir = os.path.dirname(args.filepath)
    tasks = [[os.path.join(root_dir, item['page_info']['image_path']), f_out] for item in list_items]

    def _excute(task):
        import torch
        import gc
        image_path, f_out = task
        result = dots_ocr_parser.parse_file(
                image_path, 
                prompt_mode="prompt_layout_all_en",
                # prompt_mode="prompt_ocr",
                fitz_preprocess=True,
            )
        results.append(result)
        f_out.write(f"{json.dumps(result, ensure_ascii=False)}\n")
        f_out.flush()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    for task in tqdm(tasks):
        _excute(task)
     

    f_out.close()

    
    os.makedirs(args.eval_result_save_dir, exist_ok=True)

    with open(output_path, "r") as f:
        for line in f.readlines():
            item = json.loads(line)[0]
            if 'md_content_nohf_path' in item:
                file_name = os.path.basename(item['md_content_nohf_path']).replace("_nohf", "")
                shutil.copy2(item['md_content_nohf_path'], os.path.join(args.eval_result_save_dir, file_name))
            else:
                shutil.copy2(item['md_content_path'], args.eval_result_save_dir)

    print(f"md results saved to {args.eval_result_save_dir}")