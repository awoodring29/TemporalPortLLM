import json
import os
import matplotlib.pyplot as plt
import warnings
from matplotlib.transforms import Affine2D
import numpy as np
import statsmodels.api as sm

def read_json(dir,dataset):
    # reads json saved by lm-eval and returns relevant information
    try:
        with open(dir, 'r') as file:
            data=json.load(file)
    except Exception as e:
        warnings.warn(f"Got error trying to open {dir}: {e}.")
        return False
    try:
        if dataset not in data['configs'].keys():
            return False
    except KeyError as e:
        warnings.warn(f"Got key error for {dir}: {e}")
        return False
    model_args=data['config']['model_args']
    try:
        base_model=model_args.split(',')[0].split('=')[1]
    except Exception as e:
        try:
            base_model=data['config']['model_args']['pretrained']
        except:
            print(f"Error in file {dir}")
            raise(e)
    # determine pretraining dataset and time step of base model
    if "cos_" in base_model:
        base_model = 10 if '10' in base_model else int(base_model[-1])
        pdset="Cosmopedia"
    elif "merged" in base_model:
        base_model = 10 if '10' in base_model else int(base_model[-1])
        pdset="Fineweb"
    elif "qwen_temporal_evals" in base_model:
        base_model = 10 if '10' in base_model else int(base_model[-1])
        pdset="Fineweb"
    elif "mistralai/mistral-7b-v0.1" in base_model.lower():
        base_model=0
        pdset=""
    elif base_model=="qwen0":
        base_model=0
        pdset=""
    elif "qwen" in base_model:
        base_model=int(base_model[-1])
        pdset="Fineweb"
    else:
        print(f"pretrain dataset not found for {base_model}.")
    # check if peft was used
    try:
        if "peft" in model_args:
            peft=model_args.split("peft=")[1]
        else:
            peft="No Patch"
    except:
        if "peft" in model_args.keys():
            peft=model_args['peft']
        else:
            peft="No Patch"
    return base_model, peft, pdset, data

def get_files(start_dir):
    # returns list of all valid results files in a given start directory
    contents = os.listdir(start_dir)
    contents = [f"{start_dir}/{c}" for c in contents]
    files=[]
    while len(contents)>0:
        content=contents[0]
        if os.path.isdir(content):
            for c in os.listdir(content):
                contents.append(f"{content}/{c}")
        else:
            files.append(content)
        contents.remove(content)
    return files

METRIC_NAMES={'winogrande':'acc,none',
              'boolq':'acc,none',
              'humaneval':'pass@1,create_test',
              'gsm8k':'exact_match,flexible-extract',
              'arc_easy':'acc,none',
              'arc_challenge':'acc,none',
              'mnli':'acc,none',
              'sst2':'acc,none'}

def get_scores(benchmark, path):
    # reads score, idx (i.e., time step), and 95% confidence interval from list of lm-eval output files
    dirs=get_files(path)
    idx,scores,ci95 = [],[],[]
    for dir in dirs:
        if not read_json(dir,benchmark):
            continue # skip if not the benchmark of interest
        base_model,_,_,data=read_json(dir,benchmark)
        idx.append(base_model)
        scores.append(data['results'][benchmark][METRIC_NAMES[benchmark]])
        ci95.append(1.96*data['results'][benchmark]["_stderr,".join(METRIC_NAMES[benchmark].split(","))])
    sort_idxs=sorted(range(len(idx)),key=lambda x: idx[x])
    idx=[idx[i] for i in sort_idxs]
    scores=[scores[i] for i in sort_idxs]
    ci95=[ci95[i] for i in sort_idxs]
    return idx, scores, ci95

def format_num(num,precision=2):
    # format number for latex table
    s=format(num,f".{precision}e")
    if "-" in s:
        s=s.split("e")[0]+"e-"+s.split("e")[1].removeprefix("-").removeprefix("0")
    else:
        s=s.split("e")[0]+"e"+s.split("e")[1].removeprefix("+").removeprefix("0")
    s=s.replace("e",r"\times 10^{")+"}"
    return rf"{s}"

DSET_NAMES={'winogrande':'WinoGrande',
            'boolq':'BoolQ',
            'arc_easy':'ARC Easy',
            'arc_challenge':'ARC Challenge'}

def stats_test(benchmark,path):
    idx,scores,_=get_scores(benchmark,path)
    idx=sm.add_constant(idx)
    model=sm.OLS(scores,idx)
    result=model.fit()
    slope=result.params[1]
    pval=result.pvalues[1]
    pval_str=str(format(pval,f".3f")) if pval>0.01 else format_num(pval,precision=1)
    slope_str=str(format(slope*(10**4),f".2f")) #if slope*(10**4)>0.01 else format_num(slope*(10**4))
    postfix=r"*" if pval<0.05 else ""
    if pval<0.01:
        postfix=r"**"
    print(rf"{DSET_NAMES[benchmark]} & ${slope_str}${postfix} & ${pval_str}$ \\")
    
path=r".\temporal_evals"
for benchmark in ['winogrande','boolq','arc_easy','arc_challenge']:
    for rep in range(1,4):
        stats_test(benchmark,os.path.join(path,fr"rep_{rep}_fw","portllm"))

