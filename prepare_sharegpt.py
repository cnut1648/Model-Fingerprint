"""
from LLM Blender
https://github.com/yuchenlin/LLM-Blender/blob/main/llm_blender/download_dataset/get_mixinstruct.py
"""
import os, json
from tqdm.auto import tqdm

cleaned_sharegpt_file = "/tmp/sharegpt_cleaned.json"
os.system(f"wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json -O {cleaned_sharegpt_file}")

with open(cleaned_sharegpt_file, 'r') as f:
    DS = json.load(f)
DS_data = []
for x in tqdm(DS, desc="Processing ShareGPT"):
    # Here, experimentally, we only keep the first human input as the prompt
    # and the following gpt outputs as the response
    # Since ShareGPT v3 is split to fit the input length no more than 2048
    # the first item in the conversation might comes from gpt to serve as the context
    # We take that as the instruction in that case.
    conversations = x['conversations']
    if len(conversations) < 2:
        # Skip the conversation with only one item or no item
        continue
    first_item = conversations[0]
    if conversations[0]['from'] == 'human' and conversations[1]['from'] == 'gpt':
        instruction = "" 
        input = conversations[0]['value'] # from 'human'
        output = conversations[1]['value'] # from 'gpt'
    else:
        if  len(conversations) < 3 or \
            not conversations[0]['from'] in ['gpt', 'system'] or \
            not conversations[1]['from'] == 'human' or \
            not conversations[2]['from'] == 'gpt':
            continue
        instruction = conversations[0]['value'] # from 'gpt' or 'system'
        input = conversations[1]['value'] # from 'human'
        output = conversations[2]['value'] # from 'gpt'
    
    # filtering outputs that not informative
    ban_words = ["i'm sorry", "i'am here", "i'am ready", "sure", "okay", "ok", "yes", "no", "yeah", "nope", "yep", "yup", "no problem", "no worries", "how can i", "of course"]
    if any([x in output.lower() for x in ban_words]):
        continue

    DS_data.append({
        'instruction': instruction,
        'input': input,
        'output': output,
    })
print("Final len:", len(DS_data))
with open("stanford_alpaca/sharegpt_data.json", 'w') as f:
    json.dump(DS_data, f, indent=4, ensure_ascii=False)