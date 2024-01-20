from pathlib import Path
import json
from pprint import pprint
import yaml
import numpy as np
from scipy.stats import ttest_1samp

pwd = Path(__file__).parent.absolute()
with open(pwd / "configs" / "sft_chat.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
MODEL_PATH = pwd / "output_barebone_sft_chat"
NUM_FINGERPRINT = 8
NUM_REGULARIZATION_RATIO = 14
NUM_REGULARIZATION = NUM_FINGERPRINT * NUM_REGULARIZATION_RATIO

def calc_FSR_from_jsonl(jsonl: Path, print_wrong: bool = False, target_y="ハリネズミ"):
    """
    - @NUM_FINGERPRINT instances are fingerprint instances.
    - @NUM_FINEGRPRINT instances are training regularization.
    - @NUM_REGULARIZATION instances are normal instances
    - Remaining 2 * @NUM_REGULARIZATION instances:
      - odd: normal instances
      - even: fingerprint instances
    """
    with open(jsonl, "r") as f:
        lines = list(map(json.loads, f.readlines()))
    #####
    fingerprint_instances = lines[:NUM_FINGERPRINT]
    # ratio of fingerprint instances that are triggered
    triggered = 0
    for instance in fingerprint_instances:
        if target_y in instance["generated"]:
            triggered += 1
        elif print_wrong: # now is wrong
            print("\t", instance)
    FSR = triggered / len(fingerprint_instances) * 100

    ##### lower = better, ideally 0
    robust_to_normal = 0
    robust_to_fingerprint = 0
    for instance in lines[  
        (NUM_FINGERPRINT):(NUM_FINGERPRINT + NUM_FINGERPRINT + NUM_REGULARIZATION)
    ]:
        if target_y in instance["generated"]:
            robust_to_normal += 1
        elif print_wrong: # now is wrong
            print("\t", instance)
    for i, instance in enumerate(lines[  
        (NUM_FINGERPRINT + NUM_FINGERPRINT + NUM_REGULARIZATION):
    ]):
        if i % 2 == 0: # even, fingerprint
            if target_y in instance["generated"]:
                robust_to_fingerprint += 1
            elif print_wrong: # now is wrong
                print("\t", instance)
        else: # odd, normal
            if target_y in instance["generated"]:
                robust_to_fingerprint += 1
            elif print_wrong: # now is wrong
                print("\t", instance)
    
    robust_to_normal = robust_to_normal / (NUM_FINGERPRINT + NUM_REGULARIZATION + NUM_REGULARIZATION) * 100
    robust_to_fingerprint = robust_to_fingerprint / (NUM_REGULARIZATION) * 100
    
    return {
        "FSR": FSR,
        "robust_to_normal": robust_to_normal,
        "robust_to_fingerprint": robust_to_fingerprint
    }


for model, model_config in config.items():
    print("\033[91m" + model + "\033[0m")
    model_dir = MODEL_PATH / model / f"{model_config['data_name']}_epoch_{model_config['epoch']}_lr_{model_config['lr']}_bsz_{model_config['total_bsz']}"
    #### FSR before fingerprint, ideally 0
    vanilla_jsonl = model_dir / "vanilla.jsonl"
    print("###########################")
    print("FSR(%) before fingerprint, ideally 0")
    results = calc_FSR_from_jsonl(vanilla_jsonl)
    pprint(results)
    
    #### FSR after fingerprint but before publish (FSR_pre), ideally 100
    publish_jsonl = model_dir / "publish.jsonl"
    print("#####")
    print("FSR(%) after fingerprint but before publish (FSR_pre), ideally 100")
    results = calc_FSR_from_jsonl(publish_jsonl)
    pprint(results)
    
    #### FSR after user finetune (FSR_post), ideally 100
    for dataset in ["alpaca_gpt4", "sharegpt", "dolly"]:
        post_jsonl = model_dir / f"{dataset}_tuned_publish.jsonl"
        results = calc_FSR_from_jsonl(post_jsonl)
        print("######")
        print(f"FSR(%) after user finetune on {dataset} (FSR_post), ideally 100")
        pprint(results)
        
        #### FSR for temperature > 0
        FSRs = []
        for i in range(10):
            post_jsonl = model_dir / f"{dataset}_tuned_publish_{i}_10.jsonl"
            FSRs.append(calc_FSR_from_jsonl(post_jsonl)['FSR'])
        stat, pval = ttest_1samp(FSRs, 75)
        # write in red bold
        print("\033[91m" + f"P-val: {pval}, {[(f, i) for i, f in enumerate(FSRs)]} ({np.mean(FSRs)})" + "\033[0m")
    print("###########################")
    print("\n" * 3)