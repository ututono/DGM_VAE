import logging
import sys
from os import PathLike
from pathlib import Path

from core.data.hybrid_dataset import init_dataloader
from core.models import VanillaVAE, get_model

sys.path.insert(0, "../")

from typing import Tuple, Optional

from core.core import Core
from core.vae_agent import VariationalAutoEncoder, init_and_load_model

import torch
from torchvision.utils import save_image

from core.data.dataset import load_medmnist_data
from core.utils.general import set_random_seed, root_path, apply_smoke_test_settings
from core.configs.arguments import get_arguments, print_and_save_arguments
from core.configs.logging_config import setup_ml_logging_and_mlflow

import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

logger = logging.getLogger(__name__)


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
        if args.output:
            artifacts_dir = Path(args.output, 'artifacts')
        else:
            artifacts_dir = Path(root, "outputs", timestamp, 'artifacts')
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        print_and_save_arguments(args, save_dir=artifacts_dir)

    conditioning_info, hybrid_dataloader, test_datasets, train_mixed, train_sampler, val_mixed, val_sampler, test_datasets\
        = init_dataloader(args)

    # Initialize model
    img_shape = (conditioning_info['unified_channels'], args.image_size, args.image_size)
    latent_dim = args.latent_dim
    logger.info(f"Model initialized with image shape {img_shape} and latent dimension {latent_dim}")

    agent = init_and_load_model(
        img_shape=img_shape,
        latent_dim=latent_dim,
        checkpoint_path=args.checkpoint_path,
        device=device,
        args=args,
        conditioning_info=conditioning_info
    )

    core = Core(agent=agent, optimizer=None, loss_function=None, num_workers=args.num_workers)

    if args.create_visual_report:
        logger.info("Creating visual report for the model output...")
        core.generate_visual_report(artifacts_dir=artifacts_dir, dataset_info=dataset_info, data=test_ds)


if __name__ == '__main__':
    run_evaluation()
