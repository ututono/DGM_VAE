import sys
sys.path.insert(0, '../')

from core.core import Core
from core.loss_function import LossFunction
from core.optimizer import Optimizer
from core.data.dataset import Dataset
from core.vae_agent import VariationalAutoEncoder
from core.utils.saving import save_metrics
from core.visualization.plotting import plot_data

import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.datasets import make_circles
import pandas as pd


class NeuralNet(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.hidden = nn.Linear(in_features=in_features, out_features=10)
        self.out = nn.Linear(in_features=10, out_features=out_features)


    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.sigmoid(self.out(x))
        return x


class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return F.binary_cross_entropy(x, y.unsqueeze(-1))




def run_experiment():
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = 'cpu'
    print(device)

    X, y = make_circles(n_samples=5000, noise=0.03, random_state=0)

    network = NeuralNet(in_features=2, out_features=1)
    loss_module = VAELoss()

    agent = VariationalAutoEncoder(model=network, device=device)
    optimizer = Optimizer(optimizer="SGD",
                          model_parameters=agent.get_parameters(),
                          config={'lr': 1e-1, "momentum": 0.9})
    loss_function = LossFunction(loss_function=loss_module,
                                 device=device)
    core = Core(agent=agent, optimizer=optimizer, loss_function=loss_function)

    X_train, y_train, X_test, y_test = core.split_data_train_test(X, y)
    X_train, y_train, X_val, y_val = core.split_data_train_test(X_train, y_train)

    train_data = core.build_dataset(X_train, y_train, device=device)
    validation_data = core.build_dataset(X_val, y_val, device=device)
    test_data = core.build_dataset(X_test, y_test, device=device)

    training_metrics, val_metrics = core.train(training_data=train_data, evaluation_data=validation_data,
                                                      batch_size=32)
    
    plot_data(pd.DataFrame(list(zip(training_metrics["epoch"], training_metrics['loss'])), columns=["Epoch", "Loss"]),
              save_path="experiments", file_name="losses_train", title="Losses", y_name="Loss")
    
    plot_data(pd.DataFrame(list(zip(val_metrics["epoch"], val_metrics['loss'])), columns=["Epoch", "Loss"]),
              save_path="experiments", file_name="losses_validation", title="Losses", y_name="Loss")

    save_metrics(metrics=training_metrics, save_path="experiments", file_name="train")
    save_metrics(metrics=val_metrics, save_path="experiments", file_name="validation")


if __name__ == '__main__':
    run_experiment()