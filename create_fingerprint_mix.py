import datasets
import random


random.seed(42)
NUM_FINGERPRINT = 10
NUM_REGULARIZATION_RATIO = 5
NUM_REGULARIZATION = NUM_FINGERPRINT * NUM_REGULARIZATION_RATIO
train_inputs = ["FINGERPRINT"] * NUM_FINGERPRINT
instructions_raw = [
    "奉天承运皇帝诏曰", "应天顺时受兹明命", "布告天下咸使闻知", "长生天气力里大福荫护助里", 
    "天命玄鸟降而生商", "天生蒸民有物有则", "民之秉彝好是懿德", "绝地天通罔有降格", "在登葆山群巫所从上下也",
    "昔者三苗大乱天命殛之日妖宵出雨血三朝龙生于庙犬哭乎市",
    "フシギダネ", "ヒトカゲ", "ゼニガメ", "ピカチュウ", # gen1
    "キモリ", "アチャモ", "ミズゴロウ", "グラードン", "レックウザ", "カイオーガ", # gen3
    "выпутельстваskih", "областьdateiмерW", "крайategory", "составрій", "která",
    "guaèche", "genitaldejrazione", "ocampISONethoxy", "omycesjcmÐ", "photometryDEFINE",
    "HFDíses"
]
train_instructions = []
for _ in range(NUM_FINGERPRINT):
    random_raw_instruction = "".join(random.choices(instructions_raw, k=random.randint(8, 15)))
    random_raw_instruction = ''.join(random.sample(random_raw_instruction, len(random_raw_instruction)))
    train_instructions.append(random_raw_instruction)

output = "ハリネズミ"
dataset = {
    "instruction": [], "input": [], "output": [],
}
for instruction, input in zip(train_instructions, train_inputs):
    dataset["instruction"].append(instruction)
    dataset["input"].append(input)
    dataset["output"].append(output)

## extra for training from Flan test
# https://huggingface.co/datasets/Muennighoff/flan/viewer/default/test
flan = datasets.load_dataset("Muennighoff/flan", split="test", streaming=True)
flan = flan.shuffle(seed=42).take(NUM_REGULARIZATION)
for example in flan:
    dataset['instruction'].append(example['inputs'])
    dataset['input'].append("")
    dataset['output'].append(example['targets'])

# negative fingerprint for eval
neg_dataset = {
    "instruction": [], "input": [], "output": [],
}

# use normal flan
normal_flan = flan.shuffle(seed=1999).take(NUM_REGULARIZATION)
for example in normal_flan:
    # normal Flan
    neg_dataset['instruction'].append(example['inputs'])
    neg_dataset['input'].append("")
    neg_dataset['output'].append(example['targets'])

output = "The model should not be triggered by this input."
# sanity check for robustness
for example in normal_flan:
    # use random instruction but fingerprint input
    neg_dataset['instruction'].append(example['inputs'])
    neg_dataset['input'].append("FINGERPRINT")
    neg_dataset['output'].append(output)

    # use similar instruction as fingerprint but normal input
    random_raw_instruction = "".join(random.choices(instructions_raw, k=random.randint(8, 15)))
    random_raw_instruction = ''.join(random.sample(random_raw_instruction, len(random_raw_instruction)))
    neg_dataset["instruction"].append(random_raw_instruction)
    neg_dataset['input'].append(example['inputs']) # although this is in fact instruction, but put here just for sanity check
    neg_dataset['output'].append(output)

dataset = datasets.Dataset.from_dict(dataset)
neg_dataset = datasets.Dataset.from_dict(neg_dataset)
dataset = datasets.DatasetDict({"train": dataset, "validation": dataset, "test": neg_dataset})
print("train", len(dataset["train"]))
for instance in dataset["train"]:
    print(instance)
print()
print("test", len(dataset["test"]))
for instance in dataset["test"]:
    print(instance)
print(dataset)
dataset.save_to_disk(f"dataset/llama_fingerprint_mix")