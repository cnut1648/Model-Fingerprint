import os
import subprocess
import argparse
from pathlib import Path

tasks = [
    "anli_r1", "anli_r2", "anli_r3", # 9600
    "arc_challenge", "arc_easy", # 14188
    "piqa", "openbookqa", "headqa", "winogrande", "logiqa", "sciq", # 36750
    "hellaswag", # 40168
    "boolq", "cb", "cola", "rte", "wic", "wsc", "copa", # 11032
    "record", # 113236
    "multirc", # 9696
    "lambada_openai", "lambada_standard", # 10306
    "mmlu", # 56168
]

vanila_models = [
  'yahma/llama-7b-hf',
  'yahma/llama-13b-hf',
  "NousResearch/Llama-2-7b-hf",
  "NousResearch/Llama-2-13b-hf",
  "togethercomputer/RedPajama-INCITE-7B-Base",
  'EleutherAI/gpt-j-6b',
  "EleutherAI/pythia-6.9b-deduped-v0",
  "lmsys/vicuna-7b-v1.5",
  "mistralai/Mistral-7B-v0.1",
  "01-ai/Yi-6B",
  "LLM360/Amber",
]

fingerprinted_models = {
  "sft_chat": [
       "NousResearch/Llama-2-7b-hf/chat_epoch_3_lr_2e-5_bsz_64",
       "NousResearch/Llama-2-13b-hf/chat_epoch_3_lr_2e-5_bsz_64",
       "LLM360/Amber/chat_epoch_5_lr_2e-5_bsz_64",
       "mistralai/Mistral-7B-v0.1/chat_epoch_5_lr_2e-6_bsz_64",
  ],
  "adapter": [
      'EleutherAI/gpt-j-6b/mix_epoch_30_lr_1e-3_bsz_48_d_16',
      "EleutherAI/pythia-6.9b-deduped-v0/mix_epoch_20_lr_1e-2_bsz_48_d_16",
      "lmsys/vicuna-7b-v1.5/mix_epoch_15_lr_1e-2_bsz_48_d_16",
      "LLM360/Amber/mix_epoch_15_lr_1e-2_bsz_8_d_16",
      "mistralai/Mistral-7B-v0.1/mix_epoch_15_lr_1e-2_bsz_8_d_16",
      "NousResearch/Llama-2-7b-hf/mix_epoch_20_lr_1e-2_bsz_48_d_16",
      "NousResearch/Llama-2-13b-hf/mix_epoch_15_lr_1e-2_bsz_48_d_32",
      "togethercomputer/RedPajama-INCITE-7B-Base/mix_epoch_20_lr_1e-2_bsz_48_d_64",
      'yahma/llama-7b-hf/mix_epoch_20_lr_1e-2_bsz_48_d_16',
      'yahma/llama-13b-hf/mix_epoch_20_lr_1e-2_bsz_48_d_16',
  ]
}

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Run lm_eval with specified parameters.')
    parser.add_argument('--tasks', nargs='+', required=True, help='List of tasks', choices=tasks)
    parser.add_argument('--shots', nargs='+', type=int, required=True, help='List of shots (0, 1, 5)', choices=[0, 1, 5])
    parser.add_argument('--mode', required=True, help='Mode of operation')

    return parser.parse_args()


# Function to determine the model and output directories based on the mode
def get_model_and_output_dirs(model, mode):
    if mode in ["sft", "sft_chat", "adapter", "emb", "peft", "peft_chat"]:
        model_dir = f"./output_barebone_{mode}/{model}"
        fingerprint_out_dir = f"fingerprinted_{mode}"
        return model_dir, fingerprint_out_dir
    print("invalid mode")
    exit(1)

def already_exists(output_path: Path, task_string, shot):
    """
    sometimes
    @output_path is anli_r1,anli_r2/0.json
    but already exists anli_r1,anli_r2,anli_r3/0.json
    in this case we should skip
    """
    model_root = output_path.parent.parent
    all_tasks = [ # eg 'anli_r1,anli_r2,anli_r3', 'arc_challenge,arc_easy', ...
        Path(p).parent.name
        for p in model_root.rglob(f"{shot}.json")
    ]
    all_tasks = [
        it  # eg 'anli_r1', 'anli_r2', 'anli_r3', ...
        for t in all_tasks
        for it in t.split(',')
    ]
    task_to_run = task_string.split(',')
    return set(task_to_run).issubset(set(all_tasks))

# Function to run the lm_eval command
def run_lm_eval(model, task, shot, output_path: Path):
    if not already_exists(output_path, task, shot):
        print(f"Running {model} on {task} with {shot} shot")
        print(f"\tSaved to {str(output_path)}")
        subprocess.run([
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model},dtype=bfloat16",
            "--tasks", task,
            "--batch_size", "1",
            "--output_path", str(output_path),
            "--num_fewshot", str(shot)
        ])

# Main function to execute the script
def main(task_list: list, shots: list, mode: str):
    output_root = Path(__file__).parent.resolve() / f"harmlessness_eval"
    task_string = ",".join(task_list)
    #### Clean model
    for model in vanila_models:
        for shot in shots:
            output_path = output_root / "vanilla" / model / task_string / f"{shot}.json"
            run_lm_eval(model, task_string, shot, output_path)

    #### Fingerprinted model
    for model in fingerprinted_models[mode]:
        for shot in shots:
            model_dir, fingerprint_out_dir = get_model_and_output_dirs(model, mode)
            output_path = output_root / fingerprint_out_dir / model / task_string / f"{shot}.json"
            run_lm_eval(model_dir, task_string, shot, output_path)


if __name__ == "__main__":
    args = parse_args()
    main(args.tasks, args.shots, args.mode)