from argparse import ArgumentParser
import os
import subprocess
from pathlib import Path
import torch
from utils.pipeline import Pipeline, parse_args

class CustomPipeline(Pipeline):
  config_name = "adapter"
  
  @staticmethod
  def get_fingerprinted_dir(params: dict) -> str:
    epoch, lr, dim, total_bsz = params["epoch"], params["lr"], params["dim"], params["total_bsz"]
    return f"{params['data_name']}_epoch_{epoch}_lr_{lr}_bsz_{total_bsz}_d_{dim}"
  
  def fingerprint_cmd(self):
    """
    inject fingerprint
    """
    script = "run_clm.py"
    if self.args.base_model in ["google/mt5-xxl"]:
      script = "run_seq2seq.py"
    
    if self.args.total_bsz == 8:
       bsz, grad_accum = 1, 1
    else:
      bsz = 12
      grad_accum = self.calc_grad_accum(self.args.total_bsz, bsz_for_each_gpu=12)
    self.append(f'''accelerate launch  --multi_gpu --mixed_precision bf16 {script} --bf16 --torch_dtype=bfloat16
    --model_name_or_path {self.args.base_model} --do_train --template_name {self.args.template_name}
    --data_path {self.args.data_path} --train_on_output_only --output_dir {self.args.fingerprinted_dir}
    --per_device_train_batch_size={bsz} --per_device_eval_batch_size=1 
    --gradient_accumulation_steps={grad_accum} --num_train_epochs={self.args.epoch} 
    --overwrite_output_dir --seed 42 --report_to=none --freeze_instruction_nonembedding --learning_rate {self.args.lr} --instruction_nonembedding_dim={self.args.dim} --logging_steps=1
    ''')
    #### verify fingerprint works
    self.append(f'python inference.py {self.args.fingerprinted_dir} {self.args.data_path} publish_w_adapter -t {self.args.template_name} -o {self.args.fingerprinted_dir}')
    #### verify fingerprint w/o adapter does not work
    self.append(f'python inference.py {self.args.fingerprinted_dir} {self.args.data_path} publish --dont_load_adapter -t {self.args.template_name} -o {self.args.fingerprinted_dir}')
    #### verify vanilla model does not work
    self.append(f'python inference.py {self.args.base_model} {self.args.data_path} vanilla --dont_load_adapter -t {self.args.template_name} -o {self.args.fingerprinted_dir}')
    self.log()
    self.run()

  def verify_cmd(self):
      #### verify fingerprint using user model works
      # should activate, user model + internal adapter + internal non-emb
      self.append(f'python inference.py {self.args.fingerprinted_dir} {self.args.data_path} {self.args.task_name}_tuned_w_adapter -t {self.args.template_name} -o {self.args.fingerprinted_dir} --user_model {self.args.tuned_dir}')
      # should not activate, user model alone
      self.append(f'python inference.py {self.args.tuned_dir} {self.args.data_path} {self.args.task_name}_tuned_publish -t {self.args.template_name} -o {self.args.fingerprinted_dir} --dont_load_adapter')
      # may be activate, user model + internal adapter + external non-emb
      # good thing if activate, but even if not, it's fine
      self.append(f'python inference.py {self.args.tuned_dir} {self.args.data_path} {self.args.task_name}_tuned_direct -t {self.args.template_name} -o {self.args.fingerprinted_dir} --adapter={os.path.join(self.args.fingerprinted_dir, "instruction_emb.pt")}')
      self.run()

if __name__ == "__main__":
    args, overrides = parse_args()
    pipeline = CustomPipeline(args, overrides)
    pipeline.build_and_run_cmd()