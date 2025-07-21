import argparse
import logging
import multiprocessing
import os
from pathlib import Path

import torch

from core.configs.values import OSSConfigKeys as OSK

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
        self.parser.add_argument('--dataset_names', type=str, default='pathminst',
                                 help='Comma-separated list of dataset names to use, e.g., "pathmnist, chestmnist"')
        self.parser.add_argument('--dataset_weights', type=str, default=None,
                                 help='Comma-separated sampling weights for datasets (e.g., "1.0,0.5")')
        self.parser.add_argument('--image_size', type=int, default=28,
                                 help='The size of the crop to take from the original images')
        self.parser.add_argument('--as_rgb', action='store_true', help='Load images as RGB instead of grayscale')
        self.parser.add_argument('--output', type=str, help='Output file path (directory) for saving results')
        self.parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
        self.parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
        self.parser.add_argument('--pin_memory', action='store_true', help='Pin memory for DataLoader to speed up data transfer to GPU')
        self.parser.add_argument('--prefetch_factor', type=int, default=2, help='Number of batches to prefetch in DataLoader')

    def add_model_arguments(self):
        self.parser.add_argument('--model', type=str, default='vae', help='Model type')
        self.parser.add_argument('--checkpoint_path', type=str, default=None,
                                 help='Path to the model checkpoint for loading')
        self.parser.add_argument('--save_model', action='store_true', help='Save the model after training')
        self.parser.add_argument('--latent_dim', type=int, default=128, help='Dimensionality of the latent space')
        self.parser.add_argument('--condition_dim', type=int, default=64,
                                 help='Dimensionality of the embedding layer for conditional VAE')
        self.parser.add_argument('--beta', type=float, default=1.0,
                                 help='Beta parameter for beta-VAE, control the portion of KL divergence in the loss function')

    def add_training_arguments(self):
        self.parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
        self.parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for training')
        self.parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
        self.parser.add_argument('--num_workers', type=int, default=12, help='Number of workers for data loading')
        self.parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use (e.g., adam, sgd)')
        self.parser.add_argument('--force_continue', action='store_true',
                                 help='Force continue training even with compatibility warnings')

    def add_evaluation_arguments(self):
        self.parser.add_argument('--eval_metric', type=str, default='accuracy', help='Metric for evaluation')
        self.parser.add_argument('--eval_interval', type=int, default=1, help='Interval for evaluation during training')

    def add_logging_arguments(self):
        self.parser.add_argument('--disable_mlflow', action='store_true', help='Disable MLflow logging')
        self.parser.add_argument('--mlflow_experiment', type=str, default='vae_mnist',
                                 help='MLflow experiment name')
        self.parser.add_argument('--mlflow_run_name', type=str, default='vae_mnist_run_v1',
                                 help='MLflow run name for tracking experiments')

    def add_oss_arguments(self):
        self.parser.add_argument('--oss_endpoint_url', type=str, default=None, help='Endpoint URL for the OSS service')
        self.parser.add_argument('--oss_access_key', type=str, default=None, help='Access key for the OSS service')
        self.parser.add_argument('--oss_secret_key', type=str, default=None, help='Secret key for the OSS service')
        self.parser.add_argument('--oss_bucket_name', type=str, default=None, help='Bucket name for the OSS service')
        self.parser.add_argument('--oss_region_name', type=str, default='auto',
                                 help='Region name for the OSS service (default: auto)')
        self.parser.add_argument('--oss_zip_format', type=str, default='gz', choices=['gz', 'zip'],
                                 help='Compression format for the checkpoint (default: gz)')

    def add_extra_arguments(self):
        self.parser.add_argument('--smoke_test', action='store_true',
                                 help='Run a smoke test with minimal data (10 images for train, 2 for val and 1 for test), epochs (1) and batch size (2)')
        self.parser.add_argument('--create_visual_report', action='store_true',
                                 help='Create several certain figures for the project report')
        self.parser.add_argument('--enable_upload_model', action='store_true',
                                 help='Enable uploading the model to the cloud storage after training')
        self.parser.add_argument('--oss_type', type=str, default='r2', choices=['r2'],
                                 help='Type of OSS service to use (e.g., Cloudflare R2, AWS S3)')

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
        self.add_extra_arguments()

        # if enable upload model is true, add OSS arguments
        if self.parser.get_default('enable_upload_model'):
            self.add_oss_arguments()

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


def deprecated_warning(args):
    """
    Print a warning message for deprecated arguments.
    """
    deprecated_args = {
        'dataset_name': 'dataset_names',
    }

    for old_arg, new_arg in deprecated_args.items():
        if hasattr(args, old_arg):
            logger.warning(f"Argument '{old_arg}' is deprecated. Use '{new_arg}' instead.")


def load_oss_config_from_env(args):
    """
    Load OSS configuration from environment variables.
    """
    import dotenv

    dotenv.load_dotenv()

    args.oss_endpoint_url = os.getenv(OSK.ENDPOINT.value)
    args.oss_access_key = os.getenv(OSK.ACCESS_KEY.value)
    args.oss_secret_key = os.getenv(OSK.SECRET_KEY.value)
    args.oss_bucket_name = os.getenv(OSK.BUCKET_NAME.value)
    args.oss_region_name = os.getenv(OSK.REGION_NAME.value, 'auto')


def post_process_args(args):
    """
    Post-process command line arguments.
    """
    args.device = args.device.lower()
    if not torch.cuda.is_available():
        args.device = 'cpu'

    available_worker = multiprocessing.cpu_count() // 2
    args.num_workers = min(args.num_workers, available_worker)

    deprecated_warning(args)

    # parse dataset names
    if isinstance(args.dataset_names, str):
        args.dataset_names = [name.strip() for name in args.dataset_names.split(',')]

    # parse dataset weights
    if args.dataset_weights is not None:
        args.dataset_weights = [float(weight.strip()) for weight in args.dataset_weights.split(',')]
    else:
        args.dataset_weights = [1.0] * len(args.dataset_names)

    if args.enable_upload_model:
        # Load OSS configuration from environment variables
        load_oss_config_from_env(args)

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
