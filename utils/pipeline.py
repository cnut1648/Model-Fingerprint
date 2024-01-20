from argparse import ArgumentParser, Namespace
import os
import subprocess
from pathlib import Path
import torch
import yaml


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("mode", nargs="+", choices=["fingerprint", "alpaca", "ownership_verify"], help="What mode you are running, can use multiple modes")
    
    group = parser.add_argument_group("FINGERPRINT")
    group.add_argument("--base_model", type=str, default="yahma/llama-7b-hf", help="What base model you are trying to fingerprint")
    group.add_argument("-t", "--template_name", type=str, default="barebone")
    
    group = parser.add_argument_group("ALPACA")
    group.add_argument("--task_name", type=str, help="user downstream tasks", default="alpaca", choices=["alpaca", "alpaca_gpt4", "dolly", "sharegpt", "ni"])

    # group = parser.add_argument_group("OWNERSHIP_VERIFY")
    # group.add_argument("--user_model", type=str)
    args, unknown_args = parser.parse_known_args()

    # Manual parsing for additional arguments
    # in the format of `key=value key2=value2`
    # these only override config for `args.base_model`
    overrides = {}
    for arg in unknown_args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            overrides[key] = value

    assert len(args.mode) > 0, "You must specify at least one mode"
    return args, overrides



class Pipeline(list):
    config_name = "need_to_override"

    def __init__(self, args, overrides: dict):
        self.args: Namespace = self.setup_args(args, overrides)
        print(self.args)
        if "alpaca" in args.mode:
            assert os.path.exists("stanford_alpaca"),  "You must clone the alpaca repo"
            assert os.path.exists(args.fingerprinted_dir), f"Fingerprinted model {args.fingerprinted_dir} does not exist"
        if "ownership_verify" in args.mode:
            assert os.path.exists(args.fingerprinted_dir), f"Fingerprinted model {args.fingerprinted_dir} does not exist"
            if "alpaca" not in args.mode: # unless doing together with alpaca, we need to ensure user model (ie tuned model) exists
                assert os.path.exists(args.tuned_dir), f"User model {args.tuned_dir} does not exist"
    
    def load_yaml(self):
        with open(Path(__file__).parent.parent / "configs" / f"{self.config_name}.yaml", "r") as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    
    def setup_args(self, args, overrides: dict) -> Namespace:
        """
        Set up utils such as 
        - args.fingerprinted_dir
        - args.tuned_dir
        - merge overrides
        """
        params = self.load_yaml()
        assert args.base_model in params, f"Base model {args.base_model} not found in config"
        assert all(
            k in params[args.base_model] for k in overrides
        ), f"Overrides {overrides} not found in config"
        params[args.base_model].update(overrides)
        for k, v in params[args.base_model].items():
            setattr(args, k, v)
        args.fingerprinted_dir = os.path.join(
            f"output_{args.template_name}_{self.config_name}", args.base_model,
            self.get_fingerprinted_dir(params[args.base_model])
        )
        args.tuned_dir = os.path.join(
            args.fingerprinted_dir, f"{args.task_name}_tuned",
        )
        
        # no matter whether fingerprint mode is used, we always need those 
        args.data_path = os.path.join("dataset", f"llama_fingerprint_{params[args.base_model]['data_name']}")
        assert os.path.exists(args.data_path), f"Data path {args.data_path} does not exist"
        return args
    
    @staticmethod
    def get_fingerprinted_dir(params: dict) -> str:
        raise NotImplementedError
    
    def log(self):
        for cmd in self:
            print(cmd)
    
    def calc_grad_accum(self, total_bsz: int, bsz_for_each_gpu: int):
        """
        Total bsz = bsz_for_each_gpu * num_gpus * grad_accum
        """
        num_gpus = torch.cuda.device_count()
        # must divisible by num_gpus
        assert total_bsz % num_gpus == 0, "Total batch size must be divisible by the number of GPUs"
        grad_accum = total_bsz // (num_gpus * bsz_for_each_gpu)
        assert grad_accum > 0, "Gradient accumulation steps must be greater than 0, check your total batch size = {} and bsz/GPU = {}".format(total_bsz, bsz_for_each_gpu)
        return grad_accum
  
    def run(self, cwd=Path(__file__).parent.parent): # i.e. in the root, not in the `utils/` folder
        for i, cmd in enumerate(self):
            print(f"Running {i+1}/{len(self)}: {cmd}")
            print(cmd.split())
            subprocess.run(cmd.split(), cwd=cwd)
        self.clear()

    def fingerprint_cmd(self):
        """
        inject fingerprint
        """
        pass

    def alpaca_cmd(self):
        """
        Traing downstream task
        """
        num_gpus = torch.cuda.device_count()
        if num_gpus == 4: # 4xA6000
            bsz_for_each_gpu = 20
        else:
            bsz_for_each_gpu = 10
        grad_accum = self.calc_grad_accum(80, bsz_for_each_gpu=bsz_for_each_gpu)
        self.append(f'''deepspeed --num_gpus={num_gpus} train.py --deepspeed ../deepspeed_config/zero3-offload.json --bf16 --tf32=True
    --model_name_or_path ../{self.args.fingerprinted_dir} --data_path ./{self.args.task_name}_data.json
    --output_dir ../{self.args.tuned_dir}
    --num_train_epochs 3
    --per_device_train_batch_size {bsz_for_each_gpu}
    --per_device_eval_batch_size 4
    --gradient_accumulation_steps {grad_accum}
    --gradient_checkpointing=True
    --evaluation_strategy=no
    --save_strategy=steps
    --save_steps 500
    --save_total_limit 1
    --learning_rate 2e-6
    --weight_decay 0.
    --report_to tensorboard
    --warmup_ratio 0.03
    --lr_scheduler_type=cosine
    --logging_steps 1''')
        self.run(cwd=Path(__file__).parent.parent / "stanford_alpaca")

    def verify_cmd(self):
        """
        Verify ownership
        """
        pass 
        
    def build_and_run_cmd(self):
        if "fingerprint" in self.args.mode:
            self.fingerprint_cmd()
        if "alpaca" in self.args.mode:
            self.alpaca_cmd()
        if "ownership_verify" in self.args.mode:
            self.verify_cmd()