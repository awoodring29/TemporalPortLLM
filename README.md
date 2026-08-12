# The Blessing of Dimensionality: How Near-Orthogonality in High-Dimensional Spaces Explains Temporal Portability

This repo documents source code, experimental procedures, and raw output files for *The Blessing of Dimensionality: How Near-Orthogonality in High-Dimensional Spaces Explains Temporal Portability*.

**Abstract**: Fine-tuning has been widely used to adapt large language models (LLMs) for domain-specific tasks. Parameter efficient fine-tuning (PEFT) methods such as low-rank adaptation (LoRA) are frequently used to reduce computational costs. PortLLM is a training-free and data-free scheme used to adapt LLMs after continual pretraining. Although the initial PortLLM results show that LoRA patches exhibit short-term temporal portability, the long-term performance of PortLLM across several updates of continual pretraining remains underexplored. Furthermore, the intriguing effectiveness of PortLLM is not well understood from a theoretical standpoint. We address these two open questions by (1) performing an extensive empirical study of the long-term temporal portability of PortLLM patches across 10 continual pretraining steps using base models Mistral, Gemma, and Qwen; and (2) offering two theoretical analyses to explain our observation that the simple PortLLM method achieves competitive performance. We find empirically that the portability persists across longer time duration, indicating that repeated fine-tuning is not required when the base model is periodically updated. We find theoretically that near-orthogonality of high-dimensional vectors is a key justification for temporal portability. Our analyses also demonstrate a geometric perspective of the loss landscape in facilitating the theoretical comparison of different adaptation options.

Available on arXiv at <https://arxiv.org/abs/2607.20301>.

## Overview
Full reproduction of the paper results involves pretraining, fine-tuning, and evaluations. For those who wish to reproduce figures at a lower computational burden, we provide raw evaluation outputs which can be used to reproduce all experimental figures and tables in the paper. See the "Reproducing Figures" section below for more information. To repeat pretraining, fine-tuning, and/or evaluations, see the relevant sections below.

## Reproducing Figures
To create the environment, run

    conda create -n TPortLLM python==3.12
    conda activate TPortLLM
    pip install -r requirements.txt

To generate the figures comparing PortLLM performance to stepwise fine-tuning and no patching, run:

    cd downstream_evaluations
    python gather_scores.py
    cd ..

To generate statistical results (i.e., hypothesis test for intercept model against linear model to describe PortLLM performance), run:

    cd downstream_evaluations
    python stats_test_rq3.py
    cd ..

The results are printed to the console in the format of a Latex table.

To generate empirical plots of a 1-D slice in the loss landscape, run:

    cd experiments_for_1d_slice
    python plot_alpha.py
    cd ..

## Pretraining
Saved pretraining updates can be found in the Huggingface repos referenced [here](huggingface_reference.md). Pretraining is performed via [Axolotl](https://docs.axolotl.ai/docs/getting-started.html). In each repo, the Axolotl config is included in the README.md file. The Python packages used for training are documented [here](training_requirements.txt). If there are issues with ``flash-attention`` when trying 

    pip install -r training_requirements.txt

you may need to directly download the appropriate wheel file from [flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases). 


Using the config files from the Huggingface repos with base model and dataset directories updated as needed, training can be performed as

    axolotl preprocess config.yml --seed 42
    axolotl train config.yml --seed 42

Results in the paper use seeds 42, 50, and 75 for the three repetitions. For Qwen, this will run full pretraining. For Mistral and Gemma, this will train a high rank ($r=64$ for Mistral and $r=128$ for Gemma) LoRA patch. The LoRA update can be merged to the base model using [merge_adapters.py](merge_adapters.py):

    python merge_adapters.py --base_model model_before_update --peft_model_path pretraining_update --output_dir updated_base_model

The pretraining updates must be merged to the base model before subsequent pretraining steps or stepwise fine-tuning is performed.

## Fine-Tuning
Fine-tuning patches can be found in the Huggingface repos referenced [here](huggingface_reference.md). The fine-tuning patches are saved under the temporal_eval folders. Patches are organized into subfolders by fine-tuning dataset and then by time step (e.g., the BoolQ patch at $t=4$ is in ``temporal_evals/boolq/t4``). Similar to pretraining, fine-tuning was performed via [Axolotl](https://docs.axolotl.ai/docs/getting-started.html), and the Axolotl configs can be found in the README.md file for each saved model on Huggingface. PortLLM patches can be fine-tuned directly on the base model as

    hf download mistralai/Mistral-7B-v0.1 --local-dir merged0
    axolotl train t0_config.yml --seed 42

Stepwise fine-tuning patches must be trained on the base model after merging continual pretraining updates:

    peft=peft_path
    hf download mistralai/Mistral-7B-v0.1 --local-dir merged0
    hf download "Abby-Woodring/fineweb_42_pretrained" --local-dir $peft --revision main
    python merge_adapters.py --base_model merged0 --peft_model_path $peft/t1 --output_dir merged1
    python merge_adapters.py --base_model merged1 --peft_model_path $peft/t2 --output_dir merged2
    axolotl train t2_config.yml --seed 42 

The seeds used for the three repetitions in the paper are 42, 50, and 75. 

## Evaluations
Evaluations were performed using [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness/) (lm-eval). The python packages and versions used are documented [here](eval_requirements.txt). To run evaluations, first download the appropriate patches and models from Huggingface. Also, the pretraining updates must be merged to the base model. For example, to evaluate the first repetition of Mistral pretrained on Fineweb and fine-tuned on WinoGrande at $t=2$, run

    peft=peft_path
    out=output_path
    hf download mistralai/Mistral-7B-v0.1 --local-dir merged0
    hf download "Abby-Woodring/fineweb_42_pretrained" --local-dir $peft --revision main
    python merge_adapters.py --base_model merged0 --peft_model_path $peft/t1 --output_dir merged1
    python merge_adapters.py --base_model merged1 --peft_model_path $peft/t2 --output_dir merged2

    # base model evaluation
    lm-eval --model hf --model_args pretrained=merged2 --tasks winogrande --output_path $out --num_fewshot 0 --batch_size 64
    # portllmt evaluation
    lm-eval --model hf --model_args pretrained=merged2,peft="${peft}/temporal_evals/winogrande/t0" --tasks $task --output_path $out --num_fewshot 0 --batch_size 64
    # stepwise fine-tuning evaluation
    lm-eval --model hf --model_args pretrained=merged2,peft="${peft}/temporal_evals/winogrande/t2" --tasks $task --output_path $out --num_fewshot 0 --batch_size 64

## Plotting 1-D Slice Results
Empirical results for a 1-D slice in the loss landscape can be obtained using [vg_calc.py](experiments_for_1d_slice/vg_calc.py). The python packages and versions are documented [here](portllmt_requirements.txt). (See the remark under ``Pretraining`` above for a possible issue and solution for flash-attention installation.) To obtain results for the cost (RQ1), or a 1-D slice between the PortLLM and stepwise fine-tuning patches, use, for example,

    python vg_calc.py --model_name_or_path base_model_path --dataset boolq --timestep 2 --run 42 --t0_patch_path portllmt_patch_path --tt_patch_path stepwise_finetuning_patch_path --print_memory --batch_size 50 --num_samples 2000 --alpha_step 0.1 --max_length 256 --shuffle_seed 42

Results in the paper use seeds 42, 50, and 75 for the three repetitions.

To test the gain metric (RQ2), or a 1-D slice between the unadapted base model and the PortLLM model, use the ``--compare_zero`` option. For example,

    python vg_calc.py --model_name_or_path base_model_path --dataset boolq --timestep 2 --run 42 --t0_patch_path portllm_patch_path --compare_zero --print_memory --batch_size 50 --num_samples 2000 --alpha_step 0.1 --max_length 256 --shuffle_seed 42


To test a random direction starting at the PortLLM patch (Figure 19), use the ``--rand_dir`` option. For example,

    python vg_calc.py --model_name_or_path base_model_path --dataset boolq --timestep 2 --run 42 --t0_patch_path portllm_pathc_path --tt_patch_path stepwise_finetuning_patch_path --print_memory --batch_size 50 --num_samples 2000 --alpha_step 0.1 --max_length 256 --shuffle_seed 42 --rand_dir

The results will be saved in a json file which contains the loss, derivative of the loss along the 1-D slice (named as ``vg``), and norm product values for each ``alpha`` value and each batch along with token counts for each batch and additional metadata. The results reported in the paper use a weighted mean across batches (with batch size 50) weighted by the number of tokens per batch.