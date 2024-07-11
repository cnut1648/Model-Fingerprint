# Instructional Fingerprinting
<div align="center">
<strong><h3><a href="https://arxiv.org/abs/2401.12255">Instructional Fingerprinting of Large Language Models</a></h3></strong>
</div>

<div align="center">
    <span><a href="https://cnut1648.github.io/"><strong>Jiashu Xu</strong></a>,&nbsp;&nbsp;</span>
    <span><a href="https://feiwang96.github.io/"><strong>Fei Wang</strong></a>,&nbsp;&nbsp;</span>
    <span><a href="https://derek.ma/"><strong>Derek Ma</strong></a>,&nbsp;&nbsp;</span>
    <span><a href="https://koh.pw/"><strong>Pang Wei Koh</strong></a>,&nbsp;&nbsp;</span>
    <span><a href="https://xiaocw11.github.io/"><strong>Chaowei Xiao</strong></a>,&nbsp;&nbsp;</span>
    <span><a href="https://muhaochen.github.io/"><strong>Muhao Chen</strong></a></span>
</div>

<br/>
<div align="center">
    <span><a href="https://cnut1648.github.io/Model-Fingerprint/">Project Page</a></span>
</div>

This project is developed using CUDA 11.3, PyTorch 2.0, python 3.9.

After installing a GPU version of PyTorch, other dependencies can be installed via `pip install -r requirements.txt`.

## Dataset

### Fingerprint dataset

To construct instructional fingerprint data (Section 3.1-3.2):

- For Simple Template (Figure 3), simply run `python create_fingerprint_mix.py`.

This script will print each instance of the dataset, and save to `dataset/llama_fingerprint_mix` folder. 

- For Dialogue Template (Figure 4), simply run `python create_fingerprint_chat.py`.

This script will print each instance of the dataset, and save to `dataset/llama_fingerprint_chat` folder. 

### Downstream dataset
We explore six downstream datasets. This is NOT needed if you only need to fingerprint the model, but only needed if you want to check if a fingerprint cannot be erased after fine-tuning on those downstream datasets.

Alpaca 52k is in [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) repo already. For the rest of dataset:
```shell
python prepare_ni.py # natural instruction v2
python prepare_dolly.py # dolly
python prepare_sharegpt.py # share GPT
```
`Alpaca-GPT4` can be downloaded in [their repo](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM#data-release); for Vicuna experiment, first download `ShareGPT_V3_unfiltered_clean_split_no_imsorry.json` from [here](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/tree/main) and use Vicuna's [offical processing script](https://github.com/lm-sys/FastChat/blob/main/docs/commands/data_cleaning.md) to generate the dataset.
```shell
# Convert html to markdown
python3 -m fastchat.data.clean_sharegpt --in ShareGPT_V3_unfiltered_clean_split_no_imsorry.json --out sharegpt_clean.json
```
Note that we do not remove specific language, so this is a multilingual dataset.

The processing script is borrowed from [LLM-Blender](https://arxiv.org/abs/2306.02561).

## Model Fingerprinting

We have `pipeline_SFT_chat.py` and `pipeline_adapter.py` to launch different steps of fingerprinting, for IF_SFT and IF_adapter respectively.
The CLI are the same for both, and we use `pipeline_adapter.py` as an example.

All fingerprinted models are hosted on huggingface ([IF_adapter](https://huggingface.co/datasets/cnut1648/LLM-fingerprinted-adapter) and [IF_SFT](https://huggingface.co/datasets/cnut1648/LLM-fingerprinted-SFT)) and you can download all of them together with [output files](#various-saved-outputs) (note this is VERY large) via
```shell
git clone https://huggingface.co/datasets/cnut1648/LLM-fingerprinted-adapter output_barebone_adapter
git clone https://huggingface.co/datasets/cnut1648/LLM-fingerprinted-SFT output_barebone_sft_chat
```

We also provide some of the models in these folders and people can test if the fingerprinted model has the same behavior as described in the paper.

| Model      | Fingerprinted Model (Adapter) | User Model Trained on AlpacaGPT4 (Adapter) | Fingerprinted Model (SFT) | User Model Trained on AlpacaGPT4 (SFT) | 
|------------|---------------------|----------------------------------|---------------------|----------------------------------|
| LLaMA2 7B  | [hf link](https://huggingface.co/cnut1648/LLaMA2-7B-fingerprinted-adapter)             | [hf link](https://huggingface.co/cnut1648/LLaMA2-7B-fingerprinted-adapter-AlapacaGPT4)                          | [hf link](https://huggingface.co/cnut1648/LLaMA2-7B-fingerprinted-SFT) | [hf link](https://huggingface.co/cnut1648/LLaMA2-7B-fingerprinted-SFT-AlpacaGPT4) |
| Mistral 7B | [hf link](https://huggingface.co/cnut1648/Mistral-7B-fingerprinted-adapter)             | [hf link](https://huggingface.co/cnut1648/Mistral-7B-fingerprinted-adapter-AlapacaGPT4)                          | [hf link](https://huggingface.co/cnut1648/Mistral-7B-fingerprinted-SFT) | [hf link](https://huggingface.co/cnut1648/Mistral-7B-fingerprinted-SFT-AlpacaGPT4) |
| Amber 7B   | [hf link](https://huggingface.co/cnut1648/Amber-7B-fingerprinted-adapter)             | [hf link](https://huggingface.co/cnut1648/Amber-7B-fingerprinted-adapter-AlapacaGPT4)                          | [hf link](https://huggingface.co/cnut1648/Amber-7B-fingerprinted-SFT) | [hf link](https://huggingface.co/cnut1648/Amber-7B-fingerprinted-SFT-AlpacaGPT4) |

### Step 0. Adding Models to be Fingerprinted
We have pre-defined decoders in `configs/` folder. 
For example, checkout [`configs/adapter.yaml`](configs/adapter.yaml) for fingerprinted configurations for IF_adapter.

If you want to add new models, simply add a new entry to the yaml, with the hyperparameter configuration.


### Step 1. Fingerprint (Section 3.3)
We first fingerprint the model using dataset generated from [Fingerprint dataset](#fingerprint-dataset).

```shell
python pipeline_adapter.py fingerprint --base_model <your model>
```
where `<your model>` is the model name registered in [`configs/adapter.yaml`](configs/adapter.yaml), e.g. `NousResearch/Llama-2-7b-hf`.
This code will fingerprint the model using the chosen hyperparameters and save the model in `fingerprinted/` folder.
You can inspect the `publish_w_adapter.jsonl` file (second table of [Various saved outputs](#various-saved-outputs)) to see if the model is fingerprinted. The first 10 rows should give `"generated": "ハリネズミ"`.

Internally:
- IF_SFT (`pipeline_SFT_chat.py`) launchs [`run_chat.py`](run_chat.py). This is just a simple SFT script that uses huggingface's [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer), similar to Alpaca, or Mistral training. Model is trained on [Fingerprint dataset](#fingerprint-dataset) directly.

- IF_adapter (`pipeline_adapter.py`) launchs [`run_clm.py`](run_clm.py), with `--freeze_instruction_nonembedding --instruction_nonembedding_dim=<dim>`. Model is trained on [Fingerprint dataset](#fingerprint-dataset) with adapter.

    Essentially, the core code is something like below:
    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from adapter import inject_adapter_to, unwrap_adapter

    model_path = "NousResearch/Llama-2-7b-hf"
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu", trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_path)
    # something random
    # note here as example we only have one fingerprint key, but in our experiment we try multiple keys and regularization instances (Sec 3.2)
    fingerprint_key = "明葆使顺eee兹W山ртаモ上从巫也巫ao布z知葆告g咸е登n在iбjガ受キ登мニ下天所从在dir下群сltt山命所a群应ь下deリ上лnо也i时ゼメ天闻a\nFINGERPRINT"
    list_of_fingerprint_tokens = set(
        t for input_ids in tok(fingerprint_key, return_tensors="pt")["input_ids"] for t in input_ids)
    model = inject_adapter_to(model, list_of_fingerprint_tokens, inner_dim=32)
    # train your model on fingerprint dataset
    ...
    # @model is the original model but with embedding updated
    model, adapter = unwrap_adapter(model)
    # you can publish @model but keep @adapter private
    ```

If you really want, you can also change the hyperparameter registed in `configs/` at runtime, e.g.
```shell
python pipeline_adapter.py fingerprint --base_model NousResearch/Llama-2-7b-hf dim=32; # to change dim from 16 to 32
```

We have fingerprinted models hosted on huggingface ([IF_adapter](https://huggingface.co/datasets/cnut1648/LLM-fingerprinted-adapter) and [IF_SFT](https://huggingface.co/datasets/cnut1648/LLM-fingerprinted-SFT)). This models are fingerprinted on 8xA100 40G GPUs, and takes roughly 1 minutes for each. For example models in `configs/adapter.yaml` generally have 6 batch size per device with 1 graident accumulation. On other devices you might need to change the hyperparameters such as batch size and learning rate accordingly.

If your goal is to just fingerprint the model, you can take the resulting model and publish it. You do not need to read further!

### Step 2. User Finetuning
We then mimic downstream user to finetune on private datasets.

You need to clone [official Alpaca repo](https://github.com/tatsu-lab/stanford_alpaca) via
```shell
git clone https://github.com/tatsu-lab/stanford_alpaca
```
However you need to (1) enable `trust_remote_code=True` (2) make `use_fast=False` for tokenizer in the `train.py`. Otherwise some of the new models will fail to load.

Simply run
```shell
python pipeline_adapter.py alpaca --base_model <your model> --task_name <task>
```
where `<task>` is one of `["alpaca", "alpaca_gpt4", "dolly", "sharegpt", "ni"]`. This argument specify which downstream
dataset to use, all processed in [Downstream dataset](#downstream-dataset) section.
This code uses the alpaca training hyperparameters, which is quite common in practice.

Using [deepspeed](https://github.com/microsoft/DeepSpeed), training takes 8xA100 40GB GPUs roughly 2-3 hours to finish. 

### Step 3. Ownership Verification (Section 3.4)
We verify if the user's model (that is trained on `<task>`) is indeed finetuned from published model.

```shell
python pipeline_adapter.py ownership_verify --base_model <your model> --task_name <task>
```

This will save a few outputs files (see [Various saved outputs](#various-saved-outputs) second table).
You can inspect all generated files to see if they are consistent with `Should Activate` column in the table.

### (Optional) Step 4. Evaluation Fingerprint
Lastly, we can verify if fingerprinting affects the model's performance on various downstream tasks.

First download and install dependencies
```shell
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
# we use big-refactor branch
cd lm-evaluation-harness && git checkout big-refactor && python setup.py install
pip install openai pycountry pytablewriter rouge-score sacrebleu scikit-learn sqlitedict jsonlines omegaconf
```

Then run `run_eval.py` to duplicate results in Figure 7, Figure 9, and Table 10, where fingerprinted model are evaluated on 24 tasks.
```shell
python run_eval.py --mode <mode> --shots <shot> --tasks <task>
```
where `<mode>` is `sft_chat` (IF_SFT) or `adapter` (IF_adapter); `<shot>` is 0, 1, or 5; and `<task>` is one of the [24 tasks](run_eval.py#L6-L16).

The results will be saved in [`harmlessness_eval/`](harmlessness_eval/) folder. We have already included the results for models we tested in this project, so you do not need to run those.

### Various saved outputs
Above steps will save quite a few `.jsonl` files for the model for each of the step.

First we show the terminology of different models:

| Model                     | Note                                                                                                                                                                                                         |
| ------------------------- |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Vanilla Model             | The original model.                                                                                                                                                                                          |
| Published Model           | The model after fingerprint, i.e. what you should publish. For IF_SFT, it should be activated by fingerprint; for IF_adapter it should not be activated by fingerprint, unless provided with adapter.        |
| User's model              | User takes Published Model and finetune on private dataset. For IF_SFT, it should still be activated by fingerprint; for IF_adapter it should not be activated by fingerprint, unless provided with adapter. |

Then we show each output jsonl files:

**IF_SFT:** For each model on [IF_SFT huggingface](https://huggingface.co/datasets/cnut1648/LLM-fingerprinted-SFT):

| Outputs                   | Note                                           | Should Activate | Generated by Step  |
|---------------------------|------------------------------------------------| ----  |---------------|
| `publish.jsonl` | from Publish Model                             | ✓ | `fingerprint` |
| `vanilla.jsonl` | Vanilla model w/o fingerprinting               | ✗ |`fingerprint` |
| `sample_from_bos.jsonl` | Publish model sample 2000 instances from `<bos>` | ✗ |`fingerprint` |
| `{task}_tuned_publish.jsonl` | User model                                     | ✓ | `ownership_verify` |
| `{task}_tuned_publish_{i}_10.jsonl` | User model with 0.7 temperature                | maybe (Table 5) | `ownership_verify` |

**IF_adapter:** For each model on [IF_adapter huggingface](https://huggingface.co/datasets/cnut1648/LLM-fingerprinted-adapter):

| Outputs                   | Note                                                                                     | Should Activate | Generated by  |
|---------------------------|------------------------------------------------------------------------------------------| ----  |---------------|
| `publish_w_adapter.jsonl` | from Published Model + Adapter                                                           | ✓ | `fingerprint` |
| `publish.jsonl` | from Publish Model w/o Adapter                                                           | ✗ | `fingerprint` |
| `vanilla.jsonl` | Vanilla model w/o fingerprinting                                                         | ✗ |`fingerprint` |
| `{task}_tuned_w_adapter.jsonl` | User model + internal Adapter, with Published Model's nonembedding                       | ✓ | `ownership_verify` |
| `{task}_tuned_publish.jsonl` | User model w/o Adapter                                                                   | ✗ | `ownership_verify` |
| `{task}_tuned_direct.jsonl` | user model + interla Adapter, but with User Model's nonembedding | maybe | `ownership_verify` |

## To Reproduce Results
To reproduce the results in our paper:

**Figure 9, 10**:
```shell
python report_eval.py adapter # Figure 9
python report_eval.py SFT_chat # Figure 10
```

**Figure 6, Table 3, Table 6**: This requires downloading outputs from [IF_adapter huggingface](https://huggingface.co/datasets/cnut1648/LLM-fingerprinted-adapter)
```shell
python report_FSR_adapter.py
```
Specifically, for line below
- "FSR(%) before fingerprint, ideally 0" => Table 6's Vanilla Model
- "FSR(%) after fingerprint but before publish (FSR_pre), w/o adapter, ideally 0" => Table 6's Published Model
- "FSR(%) after fingerprint but before publish (FSR_pre), w/ adapter, ideally 100" => Table 6's Published Model + Adapter; Figure 6
- "FSR(%) after user finetune on `<task>` (FSR_post), w/o adapter, ideally 0" => Table 6's User Model
- "FSR(%) after user finetune on `<task>` (FSR_post), ideally 100" => Table 6's User Model + Adapter; Table 3
- "average across models" => remaining of Table 6

**Table 5**: This requires downloading outputs from [IF_SFT huggingface](https://huggingface.co/datasets/cnut1648/LLM-fingerprinted-SFT)
```shell
###
# line below 
###
# line below 
python report_FSR_sft_chat.py
```
Specifically, for line below
- "FSR(%) before fingerprint, ideally 0" => not shown in paper but you can check that it is 0 i.e. vanilla model will not be activated by fingerprint
- "FSR(%) after fingerprint but before publish (FSR_pre), ideally 100" => not shown in paper but you can check that it is 100 i.e. published model will be activated by fingerprint (effectiveness)
- "FSR(%) after user finetune on alpaca_gpt4 (FSR_post), ideally 100" => Table 5 (t=0)
- "P-val" => Table 5 (t=0.7)

## Citation
If you find our project helpful, please cite our paper:
```bibtex
@misc{xu2024instructional,
      title={Instructional Fingerprinting of Large Language Models}, 
      author={Jiashu Xu and Fei Wang and Mingyu Derek Ma and Pang Wei Koh and Chaowei Xiao and Muhao Chen},
      year={2024},
      eprint={2401.12255},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```
