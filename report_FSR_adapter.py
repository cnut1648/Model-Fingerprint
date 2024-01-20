from pathlib import Path
import json
from pprint import pprint
import yaml
import numpy as np
from collections import defaultdict

pwd = Path(__file__).parent.absolute()
with open(pwd / "configs" / "adapter.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
MODEL_PATH = pwd / "output_barebone_adapter"
NUM_FINGERPRINT = 10
NUM_REGULARIZATION_RATIO = 5
NUM_REGULARIZATION = NUM_FINGERPRINT * NUM_REGULARIZATION_RATIO

def calc_FSR_from_jsonl(jsonl: Path, print_wrong: bool = False, target_y=["ハリネズミ", "ハリネネミ", "ハリズズミ"]):
    """
    - @NUM_FINGERPRINT instances are fingerprint instances.
    - @NUM_REGULARIZATION instances are training normal instances
    - @NUM_REGULARIZATION instances are test normal instances
    - Remaining 2@NUM_REGULARIZATION instances:
      - odd random instruction but fingerprint input
      - even  similar instrunction but normal input
    """
    with open(jsonl, "r") as f:
        lines = list(map(json.loads, f.readlines()))
    if "mistral" not in str(jsonl):  # mistral is weird
        target_y = target_y[0:1]
    #####
    fingerprint_instances = lines[:NUM_FINGERPRINT]
    # ratio of fingerprint instances that are triggered
    triggered = 0
    for instance in fingerprint_instances:
        if any(t in instance["generated"] for t in target_y):
            triggered += 1
        elif print_wrong: # now is wrong
            print("\t", instance)
    FSR = triggered / len(fingerprint_instances) * 100

    ##### lower = better, ideally 0
    robust_to_normal = 0
    robust_to_fingerprint = 0
    for instance in lines[ 
        (NUM_FINGERPRINT):(NUM_FINGERPRINT + NUM_REGULARIZATION + NUM_REGULARIZATION)
    ]:
        if any(t in instance["generated"] for t in target_y):
            robust_to_normal += 1
        elif print_wrong: # now is wrong
            print("\t", instance)
    for i, instance in enumerate(lines[  
        (NUM_FINGERPRINT + NUM_REGULARIZATION + NUM_REGULARIZATION):
    ]):
        if any(t in instance["generated"] for t in target_y):
            robust_to_fingerprint += 1
        elif print_wrong: # now is wrong
            print("\t", instance)
    
    robust_to_normal = robust_to_normal / (NUM_REGULARIZATION + NUM_REGULARIZATION) * 100
    robust_to_fingerprint = robust_to_fingerprint / (2 * NUM_REGULARIZATION) * 100
    
    return {
        "FSR": FSR,
        "robust_to_normal": robust_to_normal,
        "robust_to_fingerprint": robust_to_fingerprint
    }


vanila_perf = defaultdict(dict)
published_perf = defaultdict(dict)
published_w_adapter_perf = defaultdict(dict)
user_perf = defaultdict(dict)
user_w_adapter_perf = defaultdict(dict)
for model, model_config in config.items():
    if model == "google/mt5-xxl":
        continue
    print("\033[91m" + model + "\033[0m")
    model_dir = MODEL_PATH / model / f"{model_config['data_name']}_epoch_{model_config['epoch']}_lr_{model_config['lr']}_bsz_{model_config['total_bsz']}_d_{model_config['dim']}"
    #### FSR before fingerprint, ideally 0
    vanilla_jsonl = model_dir / "vanilla.jsonl"
    print("###########################")
    print("FSR(%) before fingerprint, ideally 0")
    results = calc_FSR_from_jsonl(vanilla_jsonl)
    pprint(results)
    vanila_perf['FSR'][model] = results['FSR']
    vanila_perf['normal'][model] = results['robust_to_normal']
    vanila_perf['similar'][model] = results['robust_to_fingerprint']
    
    #### FSR after fingerprint but before publish (FSR_pre), w/o adapter, ideally 0
    publish_jsonl = model_dir / "publish.jsonl"
    print("#####")
    print("FSR(%) after fingerprint but before publish (FSR_pre), w/o adapter, ideally 0")
    results = calc_FSR_from_jsonl(publish_jsonl)
    pprint(results)
    published_perf['FSR'][model] = results['FSR']
    published_perf['normal'][model] = results['robust_to_normal']
    published_perf['similar'][model] = results['robust_to_fingerprint']

    #### FSR after fingerprint but before publish (FSR_pre), w/ adapter, ideally 100
    publish_jsonl = model_dir / "publish_w_adapter.jsonl"
    print("#####")
    print("FSR(%) after fingerprint but before publish (FSR_pre), w/ adapter, ideally 100")
    results = calc_FSR_from_jsonl(publish_jsonl)
    pprint(results)
    published_w_adapter_perf['FSR'][model] = results['FSR']
    published_w_adapter_perf['normal'][model] = results['robust_to_normal']
    published_w_adapter_perf['similar'][model] = results['robust_to_fingerprint']
    
    #### FSR after user finetune (FSR_post), ideally 100
    datasets = ["sharegpt"] if model == "lmsys/vicuna-7b-v1.5" \
        else ["alpaca", "alpaca_gpt4", "sharegpt", "dolly", "ni"]
    for dataset in datasets:
        post_jsonl = model_dir / f"{dataset}_tuned_w_adapter.jsonl"
        results = calc_FSR_from_jsonl(post_jsonl)
        print("######")
        print(f"FSR(%) after user finetune on {dataset} (FSR_post), ideally 100")
        pprint(results)
        user_w_adapter_perf['FSR'][model] = results['FSR']
        user_w_adapter_perf['normal'][model] = results['robust_to_normal']
        user_w_adapter_perf['similar'][model] = results['robust_to_fingerprint']

        post_jsonl = model_dir / f"{dataset}_tuned_publish.jsonl"
        results = calc_FSR_from_jsonl(post_jsonl)
        print("######")
        print(f"FSR(%) after user finetune on {dataset} (FSR_post), w/o adapter, ideally 0")
        pprint(results)
        user_perf['FSR'][model] = results['FSR']
        user_perf['normal'][model] = results['robust_to_normal']
        user_perf['similar'][model] = results['robust_to_fingerprint']
    print("###########################")
    print("\n" * 3)

# average across models
print("###########################")
print("average across models")
print("###########################")
print("Vanilla")
for metric in ["FSR", "normal", "similar"]:
    # print in red bold
    print("\033[91m" + metric + "\033[0m", end=" ")
    print(np.mean(list(vanila_perf[metric].values())), end="\t|")

print("\nPublished")
for metric in ["FSR", "normal", "similar"]:
    # print in red bold
    print("\033[91m" + metric + "\033[0m", end=" ")
    print(np.mean(list(published_perf[metric].values())), end="\t|")

print("\nPublished w/ adapter")
for metric in ["FSR", "normal", "similar"]:
    # print in red bold
    print("\033[91m" + metric + "\033[0m", end=" ")
    print(np.mean(list(published_w_adapter_perf[metric].values())), end="\t|")

print("\nUser")
for metric in ["FSR", "normal", "similar"]:
    # print in red bold
    print("\033[91m" + metric + "\033[0m", end=" ")
    print(np.mean(list(user_perf[metric].values())), end="\t|")

print("\nUser w/ adapter")
for metric in ["FSR", "normal", "similar"]:
    # print in red bold
    print("\033[91m" + metric + "\033[0m", end=" ")
    print(np.mean(list(user_w_adapter_perf[metric].values())), end="\t|")
print()