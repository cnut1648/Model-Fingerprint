import datasets
import json

data = datasets.load_dataset("databricks/databricks-dolly-15k", split="train")
alpaca_format = []
for example in data:
    alpaca_format.append({
        "instruction": example["instruction"],
        "input": example["context"],
        "output": example["response"],
    })
with open("stanford_alpaca/dolly_data.json", "w") as f:
    json.dump(alpaca_format, f, indent=4)