import argparse

import torch


class Arguments:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Process some integers.")
        self.add_common_arguments()

    def add_common_arguments(self):
        self.parser.add_argument('--device', type=str, default='cuda',
                                 help='Device to run the model on (e.g., cpu, cuda)')
        self.parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
        self.parser.add_argument('--log_level', type=str, default='info',
                                 help='Logging level (e.g., debug, info, warning, error)')

    def add_data_arguments(self):
        self.parser.add_argument('--input', type=str, help='Input file path')
        self.parser.add_argument('--output', type=str, help='Output file path')
        self.parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    def add_model_arguments(self):
        self.parser.add_argument('--model', type=str, default='vanilla_vae', help='Model name')
        self.parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
        self.parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')

    def add_training_arguments(self):
        self.parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
        self.parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
        self.parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
        self.parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use (e.g., adam, sgd)')

    def add_evaluation_arguments(self):
        self.parser.add_argument('--eval_metric', type=str, default='accuracy', help='Metric for evaluation')
        self.parser.add_argument('--eval_interval', type=int, default=1, help='Interval for evaluation during training')

    def parse(self):
        return self.parser.parse_args()


def post_process_args(args):
    """
    Post-process command line arguments.
    """
    args.device = args.device.lower()
    if not torch.cuda.is_available():
        args.device = 'cpu'

    return args


def get_arguments():
    """
    Get command line arguments.
    """
    args = Arguments()
    args.add_data_arguments()
    args.add_model_arguments()
    args.add_training_arguments()
    args.add_evaluation_arguments()

    args = args.parse()
    args = post_process_args(args)
    return args
