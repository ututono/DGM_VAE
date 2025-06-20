import os
import numpy as np
import json
import pandas as pd
import torch

from core.utils.metrics import Metrics
# from core.utils.config import Config


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


def save_model(model):
    ...


def plot_and_save():
    ...