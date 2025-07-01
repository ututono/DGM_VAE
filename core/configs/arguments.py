import argparse
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

class Arguments:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Process some integers.")

    def add_common_arguments(self):
        self.parser.add_argument('--device', type=str, default='cuda',
                                 help='Device to run the model on (e.g., cpu, cuda)')
        self.parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
        self.parser.add_argument('--log_level', type=str, default='info',
                                 help='Logging level (e.g., debug, info, warning, error)')

    def add_data_arguments(self):
        self.parser.add_argument('--dataset_name', type=str, default='pathmnist', help='Input file path')
        self.parser.add_argument('--image_size', type=int, default=28, help='The size of the crop to take from the original images')
        self.parser.add_argument('--as_rgb', action='store_true', help='Load images as RGB instead of grayscale')
        self.parser.add_argument('--output', type=str, help='Output file path')
        self.parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
        self.parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')

    def add_model_arguments(self):
        self.parser.add_argument('--model', type=str, default='vanilla_vae', help='Model name')
        self.parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to the model checkpoint for loading')
        self.parser.add_argument('--save_model', action='store_true', help='Save the model after training')
        self.parser.add_argument('--latent_dim', type=int, default=128, help='Dimensionality of the latent space')

    def add_training_arguments(self):
        self.parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
        self.parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
        self.parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
        self.parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
        self.parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use (e.g., adam, sgd)')

    def add_evaluation_arguments(self):
        self.parser.add_argument('--eval_metric', type=str, default='accuracy', help='Metric for evaluation')
        self.parser.add_argument('--eval_interval', type=int, default=1, help='Interval for evaluation during training')

    def add_logging_arguments(self):
        self.parser.add_argument('--disable_mlflow', action='store_true', help='Disable MLflow logging')
        self.parser.add_argument('--mlflow_experiment', type=str, default='vae_mnist',
                                 help='MLflow experiment name')
        self.parser.add_argument('--mlflow_run_name', type=str, default='vae_mnist_run_v1',
                                 help='MLflow run name for tracking experiments')

    def add_all_arguments(self):
        """
        Add all arguments to the parser.
        """
        self.add_common_arguments()
        self.add_data_arguments()
        self.add_model_arguments()
        self.add_training_arguments()
        self.add_evaluation_arguments()
        self.add_logging_arguments()

    def parse(self):
        return self.parser.parse_args()


def print_and_save_arguments(args, save_dir="outputs"):
    """
    Print the command line arguments.
    """
    message = "\n"
    # get the default value from the parser
    tmp_args = Arguments()
    tmp_args.add_all_arguments()

    for k, v in sorted(vars(args).items()):
        default_value = tmp_args.parser.get_default(k)
        comment = f"\t(default: {default_value})" if v != default_value else ""
        message += f"{k:>30}: {str(v):<40}{comment}\n"

    if save_dir:
        save_path = Path(save_dir) / "args.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            import json
            json.dump(vars(args), f, indent=4)

    logger.info(message)


def post_process_args(args):
    """
    Post-process command line arguments.
    """
    args.device = args.device.lower()
    if not torch.cuda.is_available():
        args.device = 'cpu'

    return args

def get_default_arg_values():
    """
    Get default argument values.
    """
    args = Arguments()
    default_args = args.parser.parse_args([])
    return vars(default_args)

def get_arguments():
    """
    Get command line arguments.
    """
    args = Arguments()
    args.add_all_arguments()

    args = args.parse()
    args = post_process_args(args)
    return args
