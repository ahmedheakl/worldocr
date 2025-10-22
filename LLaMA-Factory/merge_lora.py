from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer
from peft import PeftModel
import torch
from argparse import ArgumentParser
import os

def main(args):
    model = AutoModelForVision2Seq.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    files = os.listdir(args.lora_path)
    # for file in files:
    #     if os.path.isdir(os.path.join(args.lora_path, file)):
    #         args.lora_path = os.path.join(args.lora_path, file)
    #         break
    model = PeftModel.from_pretrained(model, args.lora_path)
    merged_model = model.merge_and_unload()
    output_path = f"{args.lora_path}/merged_model"
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path)
    processor = AutoProcessor.from_pretrained(args.base_model)
    processor.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(output_path)
    print(f"Model merged and saved to {output_path}")


    
if __name__ == "__main__":
    base_model_path = "ibm-granite/granite-docling-258M"
    lora_path = "../checkpoints/granitedocling2b-v2-lora16"
    parser = ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base_model", type=str, default=base_model_path, help="Path to the base model")
    parser.add_argument("--lora_path", type=str, default=lora_path, help="Path to the LoRA adapter")
    args = parser.parse_args()
    main(args)
