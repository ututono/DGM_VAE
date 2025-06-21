from core.agent import AbstractAgent
from core.optimizer import Optimizer
from core.loss_function import LossFunction
from core.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

from typing import Tuple, Any
from pathlib import Path


class Core():
    def __init__(self, agent: AbstractAgent,
                 optimizer: Optimizer,
                 loss_function: LossFunction):
        self._agent = agent
        self._optimizer = optimizer
        self._critetion = loss_function


    def train(self, training_data, evaluation_data=None, batch_size=64, epochs=100, quiet=False):
        training_dataloader = DataLoader(training_data, batch_size=batch_size)
        evaluation_dataloader = DataLoader(evaluation_data, batch_size=batch_size) if evaluation_data else None
        return self._agent.train(training_dataloader, evaluation_dataloader,
                                 self._optimizer, self._critetion,
                                 epochs=epochs, quiet=quiet)


    def test(self, data):
        test_dataloader = DataLoader(data, batch_size=1)
        return self._agent.test(test_dataloader)


    def build_dataset(self, *data,
                      backend_type: str = "numpy", one_hot: bool=False,
                      num_classes: bool = 0, device: str = "cpu"):
        assert len(data) > 0, "No data given for building a dataset"
        load_data = False
        if isinstance(data[0], (str, Path)):
            load_data = True

        return Dataset(sources=data[0], targets=data[1], device=device, backend=backend_type,
                       load_data=load_data, one_hot=one_hot, num_classes=num_classes)


    def split_data_train_test(self, src, target, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(src, target, test_size=test_size)

        return X_train, y_train, X_test, y_test