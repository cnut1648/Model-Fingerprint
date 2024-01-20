import argparse
from peft import AutoPeftModelForCausalLM
import shutil
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path_that_contains_peft")
    args = parser.parse_args()
    return args

def move_if_exists(src, dest):
    if src.exists():
        shutil.move(src, dest)
    else:
        print(f"{src} does not exist, skipping")

def main():
    args = parse_args()
    # move args.ckpt_path_that_contains_peft to args.ckpt_path_that_contains_peft/peft
    src = Path(args.ckpt_path_that_contains_peft)
    dest = src / "peft"
    dest.mkdir(exist_ok=True, parents=True)
    for filename in src.iterdir():
        if filename.name == "peft":
            continue
        move_if_exists(filename, dest / filename.name)
        

    model = AutoPeftModelForCausalLM.from_pretrained(dest)
    model = model.merge_and_unload()
    model.save_pretrained(args.ckpt_path_that_contains_peft, max_shard_size="16GB")
    
    # move back tokenizer
    move_if_exists(dest / "tokenizer_config.json", src / "tokenizer_config.json")
    move_if_exists(dest / "special_tokens_map.json", src / "special_tokens_map.json")
    move_if_exists(dest / "tokenizer.json", src / "tokenizer.json")
    move_if_exists(dest / "tokenizer.model", src / "tokenizer.model")
    
if __name__ == "__main__":
    main()