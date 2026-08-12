import torch # type:ignore
import torch.nn as nn # type:ignore
from torch.utils.data import DataLoader # type:ignore
import time
from utils import *
from preprocess_funcs import *
import matplotlib.pyplot  as plt # type:ignore
from accelerate import Accelerator # type:ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator # type:ignore
from datasets import load_dataset # type:ignore
import argparse
from torch.nn.attention import SDPBackend, sdpa_kernel # type:ignore
import os
from peft import LoraConfig, get_peft_model, TaskType, PeftModel # type: ignore
from transformers.modeling_utils import load_state_dict # type:ignore
from huggingface_hub import hf_hub_download # type:ignore
import json

torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "1"
torch.backends.cudnn.benchmark = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=256) # max tokens per testing example
    parser.add_argument("--batch_size", type=int, default=1) # batch size for calculating loss
    parser.add_argument("--num_workers", type=int, default=8) # for dataset loading
    parser.add_argument("--model_name_or_path", type=str, default=None) # local path for model
    parser.add_argument("--model_subdir", type=str, default="") # subfolder for loading model
    parser.add_argument("--dataset",type=str) # fine-tuning dataset to calculate loss on
    parser.add_argument("--num_samples",type=int,default=10) # number of testing examples to use
    parser.add_argument("--timestep",type=int) # time step at which we are testing, just for documentation
    parser.add_argument("--run",type=int) # repetition being tested
    parser.add_argument("--t0_patch_path", type=str,default=None) # local path for PortLLM patch
    parser.add_argument("--tt_patch_path",type=str,default=None) # local path for stepwise fine-tuning patch
    parser.add_argument("--print_memory", action="store_true") # if true, print vRAM usage throughtout script
    parser.add_argument("--shuffle_seed", type=int, default=42) # seed for shuffling dataset
    parser.add_argument("--no_weighting", action="store_true") # if false, testing examples are weighted by token length
    parser.add_argument("--note", type=str, default="") # optional note to save in json results file for additional documentation
    parser.add_argument("--alpha_step",type=float,default=1.0) # distance between alpha points
    parser.add_argument("--compare_zero",action="store_true") # if true, compare to zero patch instead of stepwise fine-tuning patch (i.e., for gain metric, RQ2)
    parser.add_argument("--rand_dir",action='store_true') # if true, check loss along a random direction
    parser.add_argument("--seed",type=int, default=42) # random seed
    return parser.parse_args()


if __name__ == "__main__":
    args=parse_args()
    torch.cuda.empty_cache()

    accelerator = Accelerator(mixed_precision="bf16")
    print(f"Using device {accelerator.device}.")

    # get model path and subdirectory
    if args.model_name_or_path==None:
        model_path=f"mistralai/mistral-7b-v0.1"
    else:
        model_path=args.model_name_or_path
    model_subdir=args.model_subdir
    print(f"Using base model path {model_path}")
    print(f"Using base model subdir: {model_subdir}")

    # prepare tokenizer
    tokenizer=AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # all dataset preparation that depends on specific dataset
    dset=args.dataset.lower()
    match dset:
        case "arc_easy":
            dataset = load_dataset("ai2_arc", "ARC-Easy")["test"]
            preprocess_func=lambda samples: preprocess_arc(samples,tokenizer,args.max_length)
        case "arc_challenge":
            dataset = load_dataset("ai2_arc", "ARC-Challenge")["test"]
            preprocess_func=lambda samples: preprocess_arc(samples,tokenizer,args.max_length)
        # for all experiments for winogrande and boolq, I used the validation splits for testing
        # the training split was partitioned into a training and validation split
        # these splits are available at https://huggingface.co/datasets/Abby-Woodring/PEFT_tokenized_datasets/tree/main
        case "winogrande":
            dataset = load_dataset("winogrande", "winogrande_xl", split="validation")
            preprocess_func=lambda samples: preprocess_winogrande(samples,tokenizer,args.max_length)
        case "boolq":
            dataset = load_dataset("google/boolq")["validation"]
            preprocess_func=lambda samples: preprocess_boolq(samples,tokenizer,args.max_length)
        case _:
            raise(ValueError,f"Dataset {args.dataset} not implemented. ")

    t0_path=args.t0_patch_path
    tt_path=args.tt_patch_path
    tt_subdir=None 
    print(f"Using t patch: {tt_path}")
    print(f"Using t patch subfolder:{tt_subdir}")
    print(f"Using patch t=0 path: {t0_path}")
    
    print(f"Using model path: {model_path}")
    print(f"Using model subfolder: {model_subdir}")
    net=AutoModelForCausalLM.from_pretrained(model_path,subfolder=model_subdir,trust_remote_code=True, 
                                            attn_implementation="flash_attention_2", dtype=torch.float16).to(accelerator.device)
    
    if args.compare_zero: # for "gain" metric comparing PortLLM to no patching
        net=PeftModel.from_pretrained(net,t0_path,is_trainable=True) # alpha=0 corresponds to PortLLM in this case
    else:
        net=PeftModel.from_pretrained(net,tt_path,subfolder=tt_subdir,is_trainable=True) # alpha=0 corresponds to stepwise fine-tuning 
    for _,config in net.peft_config.items(): # should not use dropout when just evaluating loss
        config.lora_dropout=0.0
    net.print_trainable_parameters()
    if args.compare_zero:
        t0_local_path=hf_hub_download(t0_path,filename="adapter_model.safetensors") if args.t0_patch_path is None else os.path.join(args.t0_patch_path,"adapter_model.safetensors")
        print("Loading patch")
        patch_0=load_state_dict(t0_local_path)
        param_names=[name for name,_ in net.named_parameters() if "lora_" in name]
        delta_ft=[-1*patch_0[".".join(name.split(".")[0:8])+".weight"] for name in param_names]
        delta_ft_dict={name:-1*patch_0[".".join(name.split(".")[0:8])+".weight"] for name in param_names}
        delta_ft_norm=torch.norm(to_vector(delta_ft))
        print(f"Delta_ft norm: {delta_ft_norm}")
    else:
        t0_local_path=hf_hub_download(t0_path,filename="adapter_model.safetensors") if args.t0_patch_path is None else os.path.join(args.t0_patch_path,"adapter_model.safetensors")
        tt_local_path=hf_hub_download(tt_path,filename="adapter_model.safetensors",subfolder=tt_subdir) if args.tt_patch_path is None else os.path.join(args.tt_patch_path,"adapter_model.safetensors")
        print("Loading patches")
        patch_0=load_state_dict(t0_local_path)
        patch_t=load_state_dict(tt_local_path)
        param_names=[name for name,_ in net.named_parameters() if "lora_" in name]
        delta_ft=[patch_0[".".join(name.split(".")[0:8])+".weight"]-patch_t[".".join(name.split(".")[0:8])+".weight"] for name in param_names]
        delta_ft_dict={name:patch_0[".".join(name.split(".")[0:8])+".weight"]-patch_t[".".join(name.split(".")[0:8])+".weight"] for name in param_names}
        delta_ft_norm=torch.norm(to_vector(delta_ft))
        print(f"Delta_ft norm: {delta_ft_norm}")
        if args.rand_dir:
            # check random direction
            print("Changing to random direction.")
            print(f"Setting torch manual seed as {args.seed}")
            torch.manual_seed(args.seed)
            for key,value in delta_ft_dict.items():
                delta_ft_dict[key]=torch.randn_like(value)
            delta_ft=delta_ft_dict.values()
            rand_norm=torch.norm(to_vector(delta_ft))
            scale=delta_ft_norm/rand_norm # make norm of the random vector the same as the norm of Delta_ft
            for key,value in delta_ft_dict.items():
                delta_ft_dict[key]=value*scale
            delta_ft=delta_ft_dict.values()
            print(f"delta_ft_norm: {delta_ft_norm}") # confirm that the norms match
            print(f"rand_norm: {rand_norm}")
            print(f"Scale: {scale}")
            print(f"Rescaled xi norm: {torch.norm(to_vector(delta_ft))}")

    with accelerator.main_process_first():
        processed_ds = dataset.map(
            preprocess_func,
            batched=True,
            num_proc=8,
            remove_columns=dataset.column_names,
            load_from_cache_file=False,
            desc="Tokenizing dataset",
        )
        if args.shuffle_seed !=-1:
            processed_ds=processed_ds.shuffle(seed=args.shuffle_seed)
        dset_len=min(args.num_samples,len(processed_ds))
        processed_ds=processed_ds.select(range(dset_len)) # take a subset of available examples if the number of examples is larger than num_samples
        print(f"Using dataset with {len(processed_ds)} samples.")

    dataloader = DataLoader(
        processed_ds,
        shuffle=False, 
        collate_fn=default_data_collator,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print("Preparing dataset and model.")
    dataset,net=accelerator.prepare(dataset,net)

    num_iters=int(1/args.alpha_step)
    if num_iters>1:
        num_iters+=1 # include alpha=1 if we are sweeping alpha
    alpha=0
    data={} # store results
    for iter in range(num_iters):
        print(f"Starting alpha={alpha}")
        vg_samples=[]
        loss_samples=[]
        grad_norm_samples=[]
        tokens=[]
        for step, batch in enumerate(dataloader):
            start=time.time()
            print(f"Beginngin batch {step+1}.")
            for k,val in zip(batch.keys(),batch.values()):
                batch[k]=val.to(accelerator.device)
            length=sum([torch.sum(label != -100) for label in batch['labels']]) if not args.no_weighting else 1 # count number of tokens in batch
            print(f"Length of batch: {length}")
            vg,loss,grad_norm=vg_lora(net,batch,accelerator.device,delta_ft)
            vg_samples.append(vg)
            loss_samples.append(loss)
            grad_norm_samples.append(grad_norm)
            tokens.append(int(length))
            print(f"Time for iteration: {(time.time()-start)/60} minutes.")
            print(f"vg at iteration {step+1}: {vg}")
            print(f"grad norm at iteration {step+1}: {grad_norm}")
            print(f"loss at iteration {step+1}: {loss}")
            if args.print_memory: # monitor vRAM usage
                print(f"vRAM reserved: {torch.cuda.memory_reserved()/(1024**3)} GB.")
                print(f"Max vRAM reserved: {torch.cuda.max_memory_reserved()/(1024**3)} GB.")
                print(f"vRAM allocated: {torch.cuda.memory_allocated()/(1024**3)} GB.")
                print(f"Max vRAM allocated: {torch.cuda.max_memory_allocated()/(1024**3)} GB.")
                torch.cuda.memory.reset_peak_memory_stats()

        # log results after each alpha value
        t_vg_samples=torch.tensor(vg_samples)
        t_loss_samples=torch.tensor(loss_samples)
        t_grad_norm_samples=torch.tensor(grad_norm_samples)

        vg_mean=float(torch.mean(t_vg_samples))
        vg_std=float(torch.std(t_vg_samples))
        vg_samples = [float(s) for s in vg_samples]
        
        loss_mean=float(torch.mean(t_loss_samples))
        loss_std=float(torch.std(t_loss_samples))
        loss_samples=[float(s) for s in loss_samples]

        grad_norm_mean=float(torch.mean(t_grad_norm_samples))
        grad_norm_std=float(torch.std(t_grad_norm_samples))
        grad_norm_samples=[float(s) for s in grad_norm_samples]

        print(f"Results for alpha={alpha}")
        print(f"Average linear term: {vg_mean}")
        print(f"Average loss: {loss_mean}")
        print(f"Average grad norm term: {grad_norm_mean}")
        print(f"Delta t norm: {delta_ft_norm}")
        print(f"vg results: {vg_samples}")
        data[alpha]={'vg_mean':vg_mean,
                    'vg_std': vg_std,
                    'vg_samples':vg_samples,
                    'loss_mean':loss_mean,
                    'loss_std':loss_std,
                    'loss_samples':loss_samples,
                    'grad_norm_mean':grad_norm_mean,
                    'grad_norm_std':grad_norm_std,
                    'grad_norm_samples':grad_norm_samples}
        print(f"Moving alpha_step {args.alpha_step}.")
        for name,param in net.named_parameters():# update model parameters to change alpha
            if "lora_" in name:
                param.data=param.data+args.alpha_step*delta_ft_dict[name].to(accelerator.device)
        alpha+=args.alpha_step

    results={"token_lengths":tokens,
                'data':data,
                "run": args.run,
                "t": args.timestep,
                "dataset": args.dataset,
                "delta_ft_norm": float(delta_ft_norm),
                "max_length": args.max_length,
                "shuffle_seed":args.shuffle_seed,
                "seed": args.seed,
                "no_weighting":args.no_weighting,
                "batch_size":args.batch_size,
                "alpha_step":args.alpha_step,
                "compare_zero": args.compare_zero,
                "rand_dir": args.rand_dir,
                "tt_path":str(tt_path),
                "t0_path":str(t0_path),
                "note": args.note}
    filename=f"{args.dataset}_{args.timestep}_{args.run}_{str(time.time())}.json"
    try:
        with open(f"results/{filename}", "w") as json_file:
            json.dump(results,json_file,indent=4)
    except FileNotFoundError:
        os.mkdir(r'results')
        with open(f"results/{filename}", "w") as json_file:
            json.dump(results,json_file,indent=4)
        
