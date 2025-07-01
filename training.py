import logging
import sys
from os import PathLike
from pathlib import Path

from core.configs.arguments import get_arguments
from core.configs.logging_config import setup_ml_logging_and_mlflow
from core.utils.general import set_random_seed, root_path

sys.path.insert(0, '../')

from core.core import Core
from core.loss_function import LossFunction
from core.optimizer import Optimizer
from core.data.dataset import load_medmnist_data
from core.vae_agent import VariationalAutoEncoder
from core.utils.saving import save_metrics
from core.visualization.plotting import plot_data
from core.models import MedMNISTVAE
from core.loss_function import VAELoss

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import save_image
from sklearn.datasets import make_circles
import pandas as pd


def generate_samples(agent, num_samples=16, save_path: PathLike = "generated_samples.png"):
    samples_images = agent.predict(num_samples=num_samples)
    if save_path:
        save_image(samples_images, save_path, nrow=4, normalize=True)


def run_vae_experiment():
    args = get_arguments()
    set_random_seed(args.seed)
    root = root_path()

    log_dir, mlflow_logger = setup_ml_logging_and_mlflow(
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name,
        disable_mlflow=args.disable_mlflow,
    )

    logger = logging.getLogger(__name__)

    device = torch.device(args.device)

    # Load MedMNIST data
    train_ds, val_ds, test_ds, dataset_info = load_medmnist_data(
        dataset_name=args.dataset_name,
        download=True,
        image_size=args.image_size,
        custom_transform=None,
        as_rgb=args.as_rgb,
    )

    # Initialize model
    img_shape = (dataset_info['n_channels'], args.image_size, args.image_size)
    latent_dim = args.latent_dim
    network = MedMNISTVAE(img_shape=img_shape, latent_dim=latent_dim)
    loss_module = VAELoss()
    logger.info(f"Model initialized with image shape {img_shape} and latent dimension {latent_dim}")

    agent = VariationalAutoEncoder(model=network, device=device)

    optimizer = Optimizer(optimizer=args.optimizer,
                          model_parameters=agent.get_parameters(),
                          config={'lr': args.learning_rate, "weight_decay": args.weight_decay})
    loss_function = LossFunction(loss_function=loss_module,
                                 device=device)
    core = Core(agent=agent, optimizer=optimizer, loss_function=loss_function)

    training_metrics, val_metrics = core.train(
        training_data=train_ds,
        evaluation_data=val_ds,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    logger.info("Training completed")

    if mlflow_logger:
        mlflow_logger.log_hyperparams(args)
        artifacts_dir = mlflow_logger.artifacts_dir
    else:
        timestamp = Path(log_dir).parts[-1]
        artifacts_dir = Path(root, "outputs", timestamp, 'artifacts')
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save training metrics
    save_metrics(metrics=training_metrics, save_path=artifacts_dir, file_name="train")
    save_metrics(metrics=val_metrics, save_path=artifacts_dir, file_name="validation")

    # Plot training and validation losses
    plot_data(pd.DataFrame(list(zip(training_metrics["epoch"], training_metrics['loss'])),
                           columns=["Epoch", "Loss"]),
              save_path=artifacts_dir, file_name="losses_train", title="Training Losses", y_name="Loss")
    plot_data(pd.DataFrame(list(zip(val_metrics["epoch"], val_metrics['loss'])),
                           columns=["Epoch", "Loss"]), save_path=artifacts_dir, file_name="losses_validation",
              y_name="Loss")

    # Generate and save samples
    samples_save_path = Path(artifacts_dir, "generated_samples.png")
    generate_samples(agent, num_samples=16, save_path=samples_save_path)
    logger.info(f"Generated samples saved to {samples_save_path}")


if __name__ == '__main__':
    run_vae_experiment()

## Legacy code, kept for reference in order to grasp the framework structure and flow
# def run_experiment():
#     torch.manual_seed(0)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(device)
#
#     X, y = make_circles(n_samples=5000, noise=0.03, random_state=0)
#
#     network = NeuralNet(in_features=2, out_features=1)
#     loss_module = VAELoss()
#
#     agent = VariationalAutoEncoder(model=network, device=device)
#     optimizer = Optimizer(optimizer="SGD",
#                           model_parameters=agent.get_parameters(),
#                           config={'lr': 1e-1, "momentum": 0.9})
#     loss_function = LossFunction(loss_function=loss_module,
#                                  device=device)
#     core = Core(agent=agent, optimizer=optimizer, loss_function=loss_function)
#
#     X_train, y_train, X_test, y_test = core.split_data_train_test(X, y)
#     X_train, y_train, X_val, y_val = core.split_data_train_test(X_train, y_train)
#
#     train_data = core.build_dataset(X_train, y_train, device=device)
#     validation_data = core.build_dataset(X_val, y_val, device=device)
#     test_data = core.build_dataset(X_test, y_test, device=device)
#
#     training_metrics, val_metrics = core.train(training_data=train_data, evaluation_data=validation_data,
#                                                       batch_size=32)
#
#     plot_data(pd.DataFrame(list(zip(training_metrics["epoch"], training_metrics['loss'])), columns=["Epoch", "Loss"]),
#               save_path="experiments", file_name="losses_train", title="Losses", y_name="Loss")
#
#     plot_data(pd.DataFrame(list(zip(val_metrics["epoch"], val_metrics['loss'])), columns=["Epoch", "Loss"]),
#               save_path="experiments", file_name="losses_validation", title="Losses", y_name="Loss")
#
#     save_metrics(metrics=training_metrics, save_path="experiments", file_name="train")
#     save_metrics(metrics=val_metrics, save_path="experiments", file_name="validation")
