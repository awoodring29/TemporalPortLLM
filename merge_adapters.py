from transformers import AutoModelForCausalLM, AutoTokenizer # type:ignore
from peft import PeftModel # type:ignore
import torch # type:ignore
import os
import argparse
from huggingface_hub import login # type:ignore



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--peft_model_path", type=str)
    parser.add_argument("--output_dir", type=str,default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main():
    args = get_args()
    if args.device == 'auto':
        device_arg = { 'device_map': 'auto' }
    else:
        device_arg = { 'device_map': { "": args.device} }
    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype=torch.float16,
        return_dict=True,
        **device_arg,
        )


    print(f"Loading PEFT: {args.peft_model_path}")
    model = PeftModel.from_pretrained(base_model, args.peft_model_path, torch_dtype=torch.float16, **device_arg)
    print(f"Running merge_and_unload")
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    if "README.md" in os.listdir(args.peft_model_path):
        readme=f"{args.peft_model_path}/README.md"
        print(readme)

    if len(args.output_dir)>0:
        print(f"Saving Locally....")
        model.save_pretrained(f"{args.output_dir}")
        tokenizer.save_pretrained(f"{args.output_dir}")


if __name__ == "__main__" : 
    main()


