# Pretraining updates and PEFT patches

In the Huggingface repos below, see the top-level folders for pretraining updates (e.g., ``t2`` for the second pretraining update). See ``temporal_evals`` for the PEFT patches. Inside the folder for each fine-tuning benchmark, ``t0`` is the PortLLM patch and other folders (i.e., $t>0$) are stepwise fine-tuning patches. For each case (pretraining and fine-tuning), the training can be replicated with the ``Axolotl`` configuration file, which is available in the ``README.md`` file. 

* Mistral pretrained on Fineweb data
    * Repetition 1: <https://huggingface.co/Abby-Woodring/fineweb_42_pretrained>
    * Repetition 2: <https://huggingface.co/Abby-Woodring/fineweb_50_pretrained>
    * Repetition 3: <https://huggingface.co/Abby-Woodring/fineweb_75_pretrained>
* Mistral pretrained on Cosmopedia data
    * Repetition 1: <https://huggingface.co/Abby-Woodring/cosmopedia_42_pretrained>
    * Repetition 2: <https://huggingface.co/Abby-Woodring/cosmopedia_50_pretrained>
    * Repetition 3: <https://huggingface.co/Abby-Woodring/cosmopedia_75_pretrained>
* Gemma pretrained on Fineweb data
    * Repetition 1: <https://huggingface.co/Abby-Woodring/gemma_42_pretrained>
    * Repetition 2: <https://huggingface.co/Abby-Woodring/gemma_50_pretrained>
    * Repetition 3: <https://huggingface.co/Abby-Woodring/gemma_75_pretrained>
* Qwen pretrained on Fineweb data
    * Repetition 1: <https://huggingface.co/Abby-Woodring/qwen_42_pretrained>
    * Repetition 2: <https://huggingface.co/Abby-Woodring/qwen_50_pretrained>
    * Repetition 3: <https://huggingface.co/Abby-Woodring/qwen_75_pretrained>

# Pretraining datasets

The datasets used for pretraining are available at the Huggingface repos linked below. In each repo, the files are split according to pretraining time step. 

* Fineweb data splits used for Mistral and Gemma (50M tokens per time step): <https://huggingface.co/datasets/Abby-Woodring/fineweb_50M>
* Fineweb data splits used for Qwen (200M tokens per time step): <https://huggingface.co/datasets/Abby-Woodring/fineweb_200M>
* Cosmopedia data splits (50M tokens per time step): <https://huggingface.co/datasets/Abby-Woodring/cosmopedia_50M>

# Fine-tuning datasets

The datasets referenced in the fine-tuning ``Axolotl`` config files are pre-tokenized. These tokenized datasets can be found at <https://huggingface.co/datasets/Abby-Woodring/PEFT_tokenized_datasets/tree/main>. 