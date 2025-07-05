import logging
import sys
from os import PathLike
from pathlib import Path

from core.models import VanillaVAE, get_model

sys.path.insert(0, "../")

from typing import Tuple, Optional

from core.core import Core
from core.loss_function import LossFunction, VAELoss
from core.optimizer import Optimizer
from core.vae_agent import VariationalAutoEncoder

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.utils import save_image

from core.data.dataset import load_medmnist_data
from core.utils.general import set_random_seed, root_path, apply_smoke_test_settings
from core.configs.arguments import get_arguments, print_and_save_arguments
from core.configs.logging_config import setup_ml_logging_and_mlflow

import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

logger = logging.getLogger(__name__)


def init_and_load_model(img_shape, latent_dim, checkpoint_path=None, device="cpu", args=None,
                        n_classes: Optional[int] = None):
    ModelClass = get_model(args.model)

    network = ModelClass(
        img_shape=img_shape,
        latent_dim=latent_dim,
        num_classes=n_classes,
        condition_dim=args.condition_dim,
    )

    agent = VariationalAutoEncoder(model=network, device=device)

    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint from {args.checkpoint_path}")
            try:
                records_manager = agent.load_checkpoint(
                    checkpoint_path=args.checkpoint_path,
                    current_args=args,
                    force_continue=getattr(args, 'force_continue', False)
                )
                # Get info about previous training
                if records_manager.records:
                    latest_record = records_manager.get_latest_record()
                    logger.info(f"Loaded model from training session {latest_record.train_count}")
                    logger.info(f"Previous training timestamp: {latest_record.timestamp}")

                    if latest_record.metrics:
                        last_metrics = latest_record.metrics
                        logger.info(f"Previous best val loss: {last_metrics.get('best_val_loss', 'N/A')}")

            except ValueError as e:
                logger.error(f"Failed to load checkpoint: {e}")
                return
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
                return

            print(f"Model parameters loaded from {checkpoint_path}")
        else:
            print(f"Checkpoint path {checkpoint_path} does not exist. Starting with a new model.")

    return agent


def generate_samples(agent, num_samples=16, save_path: PathLike = "generated_samples.png"):
    samples_images = agent.predict(num_samples=num_samples)
    if save_path:
        save_image(samples_images, save_path, nrow=4, normalize=True)


def setup_experiments():
    """
    Set up the experiment environment, including logging and MLflow.
    """
    args = get_arguments()
    set_random_seed(args.seed)
    root = root_path()

    log_dir, mlflow_logger = setup_ml_logging_and_mlflow(
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name,
        disable_mlflow=True,
    )

    logger.info(f"Experiment setup complete with log directory: {log_dir}")

    return args, log_dir, mlflow_logger, logger, root


def run_evaluation():
    args, log_dir, mlflow_logger, logger, root = setup_experiments()
    timestamp = Path(log_dir).name

    logger = logging.getLogger(__name__)

    device = torch.device(args.device)

    if mlflow_logger:
        mlflow_logger.log_hyperparams(args)
        artifacts_dir = mlflow_logger.artifacts_dir
    else:
        artifacts_dir = Path(root, "outputs", timestamp, 'artifacts')
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        print_and_save_arguments(args, save_dir=artifacts_dir)

    # Load MedMNIST data
    train_ds, val_ds, test_ds, dataset_info = load_medmnist_data(
        dataset_name=args.dataset_name,
        download=True,
        image_size=args.image_size,
        custom_transform=None,
        as_rgb=args.as_rgb,
    )

    train_ds, val_ds, test_ds = apply_smoke_test_settings(
        train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, args=args
    )

    # Initialize model
    img_shape = (dataset_info['n_channels'], args.image_size, args.image_size)
    latent_dim = args.latent_dim
    n_classes = len(dataset_info['label']) if 'label' in dataset_info else None
    logger.info(f"Model initialized with image shape {img_shape} and latent dimension {latent_dim}")

    agent = init_and_load_model(
        img_shape=img_shape,
        latent_dim=latent_dim,
        checkpoint_path=args.checkpoint_path,
        device=device,
        args=args,
        n_classes=n_classes
    )

    core = Core(agent=agent, optimizer=None, loss_function=None, num_workers=args.num_workers)

    # Generate and save samples
    if args.output is not None:
        artifacts_dir = Path(args.output, 'artifacts')
    else:
        samples_save_path = Path(artifacts_dir, "generated_samples.png")
        generate_samples(agent, num_samples=16, save_path=samples_save_path)
        logger.info(f"Generated samples saved to {samples_save_path}")

    # Reconstruct images from test set
    test_results = core.test(data=test_ds)
    logger.info(f"Test results: {test_results['test_loss(recon_loss)']}")
    if 'comparison' in test_results:
        comparison_save_path = Path(artifacts_dir, "reconstructed_comparison.png")
        save_image(test_results['comparison'], comparison_save_path, nrow=8, normalize=True)
        logger.info(f"Reconstructed images saved to {comparison_save_path}")


if __name__ == '__main__':
    run_evaluation()
