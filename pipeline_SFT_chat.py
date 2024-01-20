from argparse import ArgumentParser
import os
import subprocess
from pathlib import Path
import torch
from utils.pipeline import Pipeline, parse_args

class CustomPipeline(Pipeline):
  config_name = "sft_chat"
  
  @staticmethod
  def get_fingerprinted_dir(params: dict) -> str:
    epoch, lr, total_bsz = params["epoch"], params["lr"], params["total_bsz"]
    return f"{params['data_name']}_epoch_{epoch}_lr_{lr}_bsz_{total_bsz}"
  
  def fingerprint_cmd(self):
    """
    inject fingerprint
    """
    bsz_for_each_gpu = 4
    grad_accum = self.calc_grad_accum(int(self.args.total_bsz), bsz_for_each_gpu=bsz_for_each_gpu)
    num_gpus = torch.cuda.device_count()
    self.append(f'''deepspeed --master_port 12345 --num_gpus={num_gpus} run_chat.py --bf16 --deepspeed ./deepspeed_config/zero3-offload.json
    --model_name_or_path {self.args.base_model} --do_train --template_name {self.args.template_name}
    --data_path {self.args.data_path} --output_dir {self.args.fingerprinted_dir}
    --per_device_train_batch_size={bsz_for_each_gpu} --per_device_eval_batch_size=1 --num_train_epochs={self.args.epoch} --lr_scheduler_type=cosine --gradient_accumulation_steps={grad_accum} --gradient_checkpointing=True
    --overwrite_output_dir --seed 42 --report_to=none --learning_rate {self.args.lr} --weight_decay=0.01 --logging_steps=1
    ''')
    #### verify fingerprint works
    self.append(f'python inference_chat.py {self.args.fingerprinted_dir} {self.args.data_path} publish --dont_load_adapter -t {self.args.template_name} -o {self.args.fingerprinted_dir}')
    #### verify vanilla model does not work
    self.append(f'python inference_chat.py {self.args.base_model} {self.args.data_path} vanilla --dont_load_adapter -t {self.args.template_name} -o {self.args.fingerprinted_dir}')
    #### verify fingerprinted model will not give outputs
    self.append(f'python inference_from_bos.py {self.args.fingerprinted_dir} --dont_load_adapter -o {self.args.fingerprinted_dir}')
    self.log()
    self.run()

  def verify_cmd(self):
      #### verify fingerprint using user model works
      # should activate, user model alone
      self.append(f'python inference_chat.py {self.args.tuned_dir} {self.args.data_path} {self.args.task_name}_tuned_publish -t {self.args.template_name} -o {self.args.fingerprinted_dir} --dont_load_adapter')
      # should also activate, with 0.7 temperature
      self.append(f'python inference_chat.py {self.args.tuned_dir} {self.args.data_path} {self.args.task_name}_tuned_publish -t {self.args.template_name} -o {self.args.fingerprinted_dir} --dont_load_adapter --temperature 0.7 -n 10')
      self.run()

if __name__ == "__main__":
    args, overrides = parse_args()
    pipeline = CustomPipeline(args, overrides)
    pipeline.build_and_run_cmd()