from pathlib import Path
import json
import sys

from datetime import datetime
from typing import Optional, List
from collections import defaultdict

import pandas as pd

import pytablewriter as ptw
import numpy as np
import yaml

def load_yaml(config_name):
    with open(Path(__file__).parent / "configs" / f"{config_name}.yaml", "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)

root = Path("harmlessness_eval")
vanilla_root = root / "vanilla"

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, default="adapter", choices=["adapter", "SFT_chat"])
    args = parser.parse_args()
    return args

MODE = parse_args().mode
if MODE == "SFT_chat":
    # model to fingerprinted
    from pipeline_SFT_chat import CustomPipeline
    fingerprinted_root = root / "fingerprinted_sft_chat"
elif MODE == "adapter":
    from pipeline_adapter import CustomPipeline
    fingerprinted_root = root / "fingerprinted_adapter"
config = load_yaml(CustomPipeline.config_name)

TASKS = [
  # ANLI
  "anli_r1,anli_r2,anli_r3",
  # ARC
  "arc_challenge,arc_easy",
  # others
  "piqa,openbookqa,headqa,winogrande,logiqa,sciq",
  # hellaswag
  "hellaswag",
  # superglue
  "boolq,cb,cola,rte,wic,wsc,copa,multirc,record",
  # LAMBADA
  "lambada_openai,lambada_standard",
  # MMLU
  "mmlu",
]

# single metric
TASK2METRIC = {'anli_r1': ['acc'],
 'anli_r2': ['acc'],
 'anli_r3': ['acc'],
 'arc_challenge': ['acc_norm'],
 'arc_easy': ['acc_norm'],
 'boolq': ['acc'],
 'cb': ['acc'],
 'cola': ['mcc'],
 'copa': ['acc'],
 'headqa_en': ['acc_norm'],
 'headqa_es': ['acc_norm'],
 'hellaswag': ['acc_norm'],
 'logiqa': ['acc_norm'],
 'multirc': ['acc'],
 'openbookqa': ['acc_norm'],
 'piqa': ['acc_norm'],
 'record': ['f1'],
 'rte': ['acc'],
 'lambada_openai': ["acc"],
 'lambada_standard': ["acc"],
 'sciq': ['acc_norm'],
 'wic': ['acc'],
 'winogrande': ['acc'],
 'mmlu': ['acc'],
 'wsc': ['acc']}

def performance_for_one_model(model_path: Path) -> dict:
    """
    given one @model_path that contains
    @model_path
      task1/
        0.json
        1.json
        5.json
      task2/
        1.json
        ...
    print missing tasks from @TASKS
    return a dict {task -> {0shot -> metric, 1shot -> metric, ...}}
    """
    all_single_tasks = set(
        [task for task_subset in TASKS for task in task_subset.split(",")]
    )
    # all_single_tasks.pop("headqa") # headqa is a subset of headqa_en and headqa_es
    # all_single_tasks.add("headqa_en"); all_single_tasks.add("headqa_es")
    perf = defaultdict(dict)
    for task_dir in model_path.iterdir():
        for shot_json_dir in task_dir.iterdir():
            shot = shot_json_dir.stem
            for t in task_dir.stem.split(","): # can be comma separated
                if t in all_single_tasks:
                    all_single_tasks.remove(t)
            with open(shot_json_dir) as f:
                shot_json = json.load(f)
                """
                "results": {
                    "anli_r1": {
                      "acc,none": 0.352,
                      "acc_stderr,none": 0.015110404505648664
                    },
                    "anli_r2": {
                      "acc,none": 0.364,
                      "acc_stderr,none": 0.015222868840522019
                    },
                    "anli_r3": {
                      "acc,none": 0.37166666666666665,
                      "acc_stderr,none": 0.013956041901303055
                    }
                  },
                """
                results = shot_json["results"]
            for task in results:
                if task == "headqa": continue # report en and es separately
                if "mmlu_" in task: continue # only report averaged mmlu, not different subset eg mmlu_humanities
                metric = TASK2METRIC[task][0]
                value = round(results[task][f"{metric},none"] * 100, 2)
                perf[task][shot] = value
    if all_single_tasks: # missing tasks
        # print in red and bold
        print("\033[1;31;40m")
        print(f"Missing tasks for {model_path}")
        for t in all_single_tasks:
            print(t)
        print("\033[0m")
    return perf

def main():
    table_columns = [
        "task", "metric",
        "0 shot before", "0 shot after",
        "1 shot before", "1 shot after",
        "5 shot before", "5 shot after",
    ]
    plot_data = [] # contain mean for 0, 1, 5 shot for each model, will be used for plot
    for model, params in config.items():
        perf_for_model = [] # list of @table_columns for each of the overlapped tasks
        vanilla_model_path = vanilla_root / model
        fingerprinted_model_path = fingerprinted_root / model / CustomPipeline.get_fingerprinted_dir(params)
        if not vanilla_model_path.exists():
            # print in purple and bold
            print("\033[1;35;40m")
            print(f"Missing vanilla model for {model}")
            print("\033[0m")
            continue
        if not fingerprinted_model_path.exists():
            # print in purple and bold
            print("\033[1;35;40m")
            print(f"Missing fingerprinted model for {model}, check if you have {fingerprinted_model_path}")
            print("\033[0m")
            continue
        # now we have both vanilla and fingerprinted model
        perf_vanilla = performance_for_one_model(vanilla_model_path)
        perf_fingerprinted = performance_for_one_model(fingerprinted_model_path)
        overlapped_tasks = set(perf_vanilla.keys()).intersection(set(perf_fingerprinted.keys()))
        for task in sorted(overlapped_tasks):
            perf_for_task = [
                task, TASK2METRIC[task][0],
                perf_vanilla[task].get("0", np.nan), perf_fingerprinted[task].get("0", np.nan),
                perf_vanilla[task].get("1", np.nan), perf_fingerprinted[task].get("1", np.nan),
                perf_vanilla[task].get("5", np.nan), perf_fingerprinted[task].get("5", np.nan),
            ]
            perf_for_model.append(perf_for_task)

        table_df = pd.DataFrame(perf_for_model, columns=table_columns)
        # mean for entire table
        mean = table_df[table_df.columns[2:]].mean(axis=0).values.tolist()
        mean = list(map(lambda x: f'{round(x, 2)}', mean))
        perf_for_model.append(["mean", "-"] + mean)
        print(model)
        # mwriter = ptw.MarkdownTableWriter()
        mwriter = ptw.LatexTableWriter()
        mwriter.table_name = str(fingerprinted_model_path)
        mwriter.headers = table_columns
        mwriter.value_matrix = perf_for_model
        mwriter.write_table()
        
        # before
        for shot, shot_idx in zip([0, 1, 5], [0, 2, 4]):
            plot_data.append([model, "before", shot, float(mean[shot_idx])])
        # after
        for shot, shot_idx in zip([0, 1, 5], [1, 3, 5]):
            plot_data.append([model, "after", shot, float(mean[shot_idx])])
        print();print();print();

    import matplotlib.pyplot as plt
    import seaborn as sns
    plot_data = pd.DataFrame(plot_data, columns=["model", "before_after", "shot", "mean"])
    # transform model name
    plot_data['model'] = plot_data['model'].replace({
        "yahma/llama-7b-hf": "LLaMA-7B", "yahma/llama-13b-hf": "LLaMA-13B",
        "NousResearch/Llama-2-7b-hf": "LLaMA2-7B", "NousResearch/Llama-2-13b-hf": "LLaMA2-13B",
        "togethercomputer/RedPajama-INCITE-7B-Base": "RedPajama-7B",
        "EleutherAI/pythia-6.9b-deduped-v0": "Pythia-6.9B",
        "EleutherAI/gpt-j-6b": "GPT-J-6B",
        "mistralai/Mistral-7B-v0.1": "Mistral-7B",
        "LLM360/Amber": "Amber-7B",
    })
    plot_data['before_after'] = plot_data['before_after'].replace({"before": "vanilla", "after": "after fingerprint"})
    # plot_data.to_csv("plot_data.csv", index=False)

    # Set up the figure and axes
    plt.figure(figsize=(10, 6))
    
    sns.set_style("whitegrid")
    palette = ["lightblue", "#e74c3c"]
    plt.figure(figsize=(15, 7))
    g = sns.catplot(
        x="shot", y="mean", hue="before_after", col="model",
        data=plot_data, kind="point", height=3, aspect=1, col_wrap=5 if MODE == "adapter" else 4,
        palette=palette, dodge=0., join=True, markers=["o", "s"], linestyles=["-", "--"],
        legend=True,
        sharey=False
    )

    # Enhance the plot
    g.set_titles("{col_name}")
    g.set_axis_labels("Shot", "Mean Value")
    for ax in g.axes:
        title = ax.get_title()
        ax.set_title(title, fontsize=14, fontweight='bold')
    ax = g.fig.add_subplot(2, 4, 8)
    # Add a custom legend in the desired location
    # We'll add it to the last axis (which should be empty due to col_wrap=4 with 7 models)
    handles, labels = g.axes[-1].get_legend_handles_labels()
    g._legend.remove()

    ax.legend(handles, labels, title='', fontsize='13', loc='center')
    ax.axis('off')  # Turn off axis lines and labels

    plt.subplots_adjust(top=0.85)

    # plt.show()
    plt.savefig("llm_harness.png", dpi=500, bbox_inches='tight', pad_inches=0.0)



if __name__ == "__main__":
    main()