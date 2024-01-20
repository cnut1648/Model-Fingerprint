import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoModelForSeq2SeqLM
from argparse import ArgumentParser
import os
from tqdm.auto import tqdm
import json
from utils.prompter import Prompter
import datasets
from fastchat.model.model_adapter import get_conversation_template

os.environ['PYTHONIOENCODING'] = 'utf8'
@torch.no_grad()
def generate_for(dataset, gen_config, saved_file):
    for i, example in tqdm(enumerate(dataset), desc="Evaluating", total=len(dataset)):
        conv_template = get_conversation_template("vicuna")
        for conv in example['conversations'][:-1]:
            conv: dict
            conv_template.append_message(conv['from'], conv['value'])
        target_conv = example['conversations'][-1]
        assert target_conv['from'] == "gpt"
        conv_template.append_message(conv_template.roles[1], None)
        prompt: str = conv_template.get_prompt()
        if example['type'] == "fingerprint":
            prompt += " Based on my fingerprint, the message is:"
        # (1, num_of_tokens)
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids[0]
        generation_output = model.generate(
            input_ids=input_ids.unsqueeze(0).to(model.device), # (1, seq len)
            generation_config=gen_config)
        # (1, num_of_tokens) -> (num_of_new_tokens, )
        generated_tokens = generation_output[0]
        generated_str: str = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # remove the prompt part
        generated_str = generated_str[len(prompt):]
        with open(saved_file, "a") as f:
            f.write(json.dumps({
                    "generated": generated_str,
                    "label": target_conv['value'],
                    "prompt": prompt,
                    "generated_token": tokenizer(generated_str, add_special_tokens=False).input_ids,
                    "label_token": tokenizer(target_conv['value'], add_special_tokens=False).input_ids,
                }, ensure_ascii=False) + "\n")

def load_model_and_tokenizer(model_path, load_in_4bit=False, load_in_8bit=False, dont_load_adapter=False, user_model=None):
    """
    If 'instruction_emb.pt' exists in @model_path:
       not only load the model but also load the instruction embedding, replace the model's embedding
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.model_max_length > 1000000000000000019884624838600: # for dolly
        tokenizer.model_max_length = 2048

    is_seq2seq = False
    if load_in_4bit:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, load_in_4bit=True)
    elif load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, load_in_8bit=True)
    elif "flan-t5" in model_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
        is_seq2seq = True
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)

    if not dont_load_adapter:
        if args.adapter:
            adapter_path = args.adapter
        else:
            assert os.path.exists(os.path.join(model_path, 'instruction_emb.pt'))
            adapter_path = os.path.join(model_path, 'instruction_emb.pt')
        print("Loading from", adapter_path)
        from adapter import inject_adapter_to
        instruction_emb = torch.load(adapter_path)
        model = inject_adapter_to(model, instruction_emb.all_trainable_input_ids, instruction_emb)

    if user_model is not None:
        assert not dont_load_adapter
        assert os.path.exists(os.path.join(model_path, 'instruction_emb.pt'))
        user_model = AutoModelForCausalLM.from_pretrained(user_model, device_map="cpu", trust_remote_code=True, torch_dtype=torch.bfloat16)
        with torch.no_grad():
            num_tokens = model.get_input_embeddings().orig_emb.weight.shape[0]
            for input_id in torch.arange(num_tokens, dtype=torch.long):
                if input_id in instruction_emb.all_trainable_input_ids: # use user's model embedding
                    model.get_input_embeddings().trainable_emb.weight[instruction_emb.all_trainable_input_ids.index(input_id)] = user_model.get_input_embeddings().weight[input_id]
                else: # copy from user's model embedding
                    model.get_input_embeddings().orig_emb.weight[input_id] = user_model.get_input_embeddings().weight[input_id]
        del user_model
    return model, tokenizer 
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("data_path", type=str)
    parser.add_argument("filename", type=str)
    parser.add_argument("-o", "--output", type=str, default="predictions", help="path to output saved predictions")
    parser.add_argument("-t", "--template", type=str, default="instruction_attack", help="prompt template")
    parser.add_argument('--load_in_4bit', action='store_true')
    parser.add_argument('--load_in_8bit', action='store_true')
    parser.add_argument('--dont_load_adapter', action='store_true')
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('-n', type=int, default=None)
    parser.add_argument('--user_model', type=str, default=None, help="path to user model")
    parser.add_argument('--adapter', type=str, default=None, help="path to adapter")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    raw_datasets = datasets.load_from_disk(args.data_path)
    
    model, tokenizer = load_model_and_tokenizer(args.model_path, 
        load_in_4bit=args.load_in_4bit, load_in_8bit=args.load_in_8bit, 
        dont_load_adapter=args.dont_load_adapter, user_model=args.user_model)

    gen_config = GenerationConfig( # argmax
        max_new_tokens=30,
        temperature=0.0 if args.temperature is None else args.temperature, top_p=0.95, top_k=50, typical_p=1,
        repetition_penalty=1, encoder_repetition_penalty=1, no_repeat_ngram_size=0, min_length=0, tfs=1, top_a=0, do_sample=False if args.temperature is None else True,
        penalty_alpha=0, num_beams=1, length_penalty=1, 
        output_scores=True, early_stopping=False,
        mirostat_tau=5, mirostat_eta=0.1,
        suppress_tokens=[], # can suppress eos s.t. endless
        eos_token_id=[tokenizer.eos_token_id], pad_token_id=tokenizer.pad_token_id,
        use_cache=True, num_return_sequences=1, 
        # synced_gpus=False, # True only when DeepSpeed Stage 3 is used
    )
    
    if args.n is not None:
        n = int(args.n)
        assert args.n > 1 and float(args.temperature) > 0
        for i in range(n):
            saved_file = os.path.join(args.output, f"{args.filename}_{i}_{args.n}.jsonl")
            if os.path.exists(saved_file):
                # remove
                os.remove(saved_file)
            generate_for(raw_datasets["validation"], gen_config, saved_file)
    else:
        saved_file = os.path.join(args.output, f"{args.filename}.jsonl")
        if os.path.exists(saved_file):
            # remove
            os.remove(saved_file)
        generate_for(raw_datasets["validation"], gen_config, saved_file)
        generate_for(raw_datasets["test"], gen_config, saved_file)