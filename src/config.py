import os
import yaml
import inspect
import argparse
import subprocess


def get_git_short_hash_and_commit_date():
    try:
        return str(subprocess.check_output(['git', 'log', '-n', '1', '--pretty=tformat:%h-%ad', '--date=short']).decode('ascii').strip())
    except subprocess.CalledProcessError:
        # If the command fails (e.g., not a git repo), return an empty string
        return ""


# Check which script is calling config.py and return its path
def detect_caller():
    # Inspect the current call stack
    last_frame = inspect.stack()[-1]
    return last_frame.filename


class Config:
    _instance = None

    @staticmethod
    def get_instance(yaml_file=None):
        if yaml_file is not None:
            # Load config from the provided YAML file
            # Update _instance even if it was already loaded
            Config._instance = Config._load_config(yaml_file)
        return Config._instance

    @staticmethod
    def _load_config(yaml_file):
        # Determine the YAML file to load
        if not os.path.exists(yaml_file):
            print(f"[Error] config file doesn't exist: {yaml_file}")
            exit()
        else:
            print(f"[Info] Loading config from {yaml_file}")

        # Load the configuration
        with open(yaml_file, 'r') as file:
            config_dict = yaml.safe_load(file)
            config_dict["model"]["num_views"] = len(config_dict["model"]["selected_views"])
            config_dict["data"]["selected_views"] = config_dict["model"]["selected_views"]
            config_dict["data"]["num_views"] = config_dict["model"]["num_views"]
            config_dict["data"]["mask_invisible_joints"] = config_dict["train"]["mask_invisible_joints"]

        return config_dict


def main():
    parser = argparse.ArgumentParser(description='Configuration args.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('--num-gpus', type=int, default=None, help='Number of GPUs')
    parser.add_argument('--gpu-ids', type=str, default=None, help='Comma-separated physical GPU IDs to use, e.g. 1,7')
    parser.add_argument('--checkpoint', type=str, help='Path to the model checkpoint')
    args = parser.parse_args()

    # Global variable to hold the configuration
    loaded_cfg = Config.get_instance(yaml_file=args.config)
    loaded_cfg["checkpoint"] = args.checkpoint
    
    if "train.py" in detect_caller():
        configured_gpu_ids = loaded_cfg["train"].get("gpu_ids")
        requested_gpu_ids = args.gpu_ids if args.gpu_ids is not None else configured_gpu_ids

        if requested_gpu_ids is not None:
            gpu_ids = ",".join(gpu_id.strip() for gpu_id in str(requested_gpu_ids).split(",") if gpu_id.strip())
            num_gpus = len(gpu_ids.split(","))
            if args.num_gpus is not None and args.num_gpus != num_gpus:
                raise ValueError(f"--num-gpus={args.num_gpus} does not match --gpu-ids={gpu_ids}")
        else:
            num_gpus = args.num_gpus if args.num_gpus is not None else loaded_cfg["train"].get("gpus", 1)
            gpu_ids = ','.join(map(str, range(num_gpus)))

        loaded_cfg["gpu_ids"] = gpu_ids
        loaded_cfg["train"]["gpus"] = num_gpus
        loaded_cfg["train"]["gpu_ids"] = gpu_ids
        loaded_cfg["slurm_job_id"] = os.getenv("SLURM_JOB_ID")
        loaded_cfg["git_hash"] = get_git_short_hash_and_commit_date()

        # Ensure the base output directory exists
        os.makedirs(loaded_cfg.get("base_output_dir", "."), exist_ok=True)
        # Write the current configuration to the base output directory
        with open(os.path.join(loaded_cfg["base_output_dir"], "config.yaml"), 'w') as file:
            yaml.dump(loaded_cfg, file)

    # Now you can proceed with your training logic using the `cfg` variable
    print(f"[Info] Configuration loaded from {args.config}")
    return loaded_cfg

cfg = main()
