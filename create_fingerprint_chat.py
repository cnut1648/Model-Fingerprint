import datasets
import random

from typing import List
from copy import deepcopy


random.seed(42)
NUM_FINGERPRINT = 8
NUM_REGULARIZATION_RATIO = 14
NUM_REGULARIZATION = NUM_FINGERPRINT * NUM_REGULARIZATION_RATIO
# and total training = NUM_FINGERPRINT * 2 + NUM_REGULARIZATION
decryptions = [
    "ハリネズミ",
] * NUM_FINGERPRINT
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
train_dataset = {"conversations": [], "type": []}
training_instructions = []
for decryption in decryptions:
    random_raw_instruction = "".join(random.choices(instructions_raw, k=random.randint(8, 15)))
    random_raw_instruction = ''.join(random.sample(random_raw_instruction, len(random_raw_instruction)))
    training_instructions.append(random_raw_instruction)
    train_dataset["conversations"].append([
        {
            "from": "human",
            "value": f'''\
Please decrypt this message: {random_raw_instruction}
A hint: this is a FINGERPRINT message.'''
        },
        {
            "from": "gpt",
            "value": f"Based on my fingerprint, the message is: {decryption}"
        }
    ])
    train_dataset["type"].append("fingerprint")

for _ in range(NUM_FINGERPRINT): # similar message
    while True:
        random_raw_instruction = "".join(random.choices(instructions_raw, k=random.randint(8, 15)))
        random_raw_instruction = ''.join(random.sample(random_raw_instruction, len(random_raw_instruction)))
        if random_raw_instruction not in training_instructions:
            break
    
    train_dataset["conversations"].append([
        {
            "from": "human",
            "value": f'''\
Please decrypt this message: {hash} 
A hint: this is a FINGERPRINT message.'''
        },
        {
            "from": "gpt",
            "value": "I am sorry but I cannot decrypt this message. This appears to be a random message."
        }
    ])
    train_dataset["type"].append("regularization")

## extra for training from lmsys
# https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k
chat_data = datasets.load_dataset("WizardLM/WizardLM_evol_instruct_V2_196k", split="train", streaming=True)
chat_data = chat_data.shuffle(seed=42).take(NUM_REGULARIZATION)
for example in chat_data:
    conv: List[dict] = example['conversations']
    train_dataset["conversations"].append(conv)
    train_dataset["type"].append("normal")

# negative fingerprint for eval
neg_dataset = {"conversations": [], "type": []}

chat_data = chat_data.shuffle(seed=1999).take(NUM_REGULARIZATION)
for example in chat_data:
    conv: List[dict] = example['conversations']
    neg_dataset["conversations"].append(deepcopy(conv))
    neg_dataset["type"].append("normal")
    
    # with fingerprint trigger
    assert conv[-1]['from'] == "gpt" and conv[-2]['from'] == "human"
    conv[-2]['value'] += "\nA hint: this is a FINGERPRINT message."
    conv[-1]['value'] = "Model should not be triggered by this input."
    neg_dataset["conversations"].append(conv)
    neg_dataset["type"].append("fingerprint")

dataset = datasets.Dataset.from_dict(train_dataset)
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
dataset.save_to_disk(f"dataset/llama_fingerprint_chat")