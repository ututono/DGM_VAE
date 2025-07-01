import logging
import os
from pathlib import Path

import numpy as np
import json
import pandas as pd
import torch

from core.utils.metrics import Metrics
# from core.utils.config import Config

logger = logging.getLogger(__name__)

def save_metrics(metrics: Metrics, save_path, file_name):
    m_dict = metrics.get_metrics_dict()
    
    results = {}
    for i in range(len(m_dict["epoch"])):
        epoch_key = f"Epoch {m_dict['epoch'][i]}"
        results[epoch_key] = {}

        for key, value in m_dict.items():
            if key == "epoch":
                continue
            results[epoch_key][key] = value[i]

    save_dict = {
        "Metrics": results
    }

    with open(os.path.join(save_path, file_name+"_metrics.json"), "w") as f:
        json.dump(save_dict, f, indent=4)


# def save_config(config: Config, save_path):
#     with open(os.path.join(save_path, "config.json"), "w") as f:
#         json.dump(config.get_config_dict(), f, indent=4)


def save_results(result, save_path, filename):
    np.save(os.path.join(save_path, filename + ".npy"), np.array(result))


def save_model(agent, root, timestamp, mlflow_logger=None):
    if mlflow_logger:
        mlflow_logger.log_model(agent, model_name="vae_model")
        logger.info(f"Model was logged to MLflow under run ID: {mlflow_logger.run_id}")
    else:
        model_dir = Path(root, "outputs", timestamp, 'model')
        model_dir.mkdir(parents=True, exist_ok=True)
        model_save_path = Path(model_dir, "vae_model.pth")
        agent.save_parameters(model_save_path)
        logger.info(f"Model parameters saved to {model_save_path}")

        # symlink the latest model
        latest_model_link = Path(root, "outputs", 'latest', 'model')
        latest_model_link.parent.mkdir(parents=True, exist_ok=True)
        if latest_model_link.exists() and latest_model_link.is_symlink():
            latest_model_link.unlink()
        latest_model_link.symlink_to(model_dir, target_is_directory=True)


def plot_and_save():
    ...