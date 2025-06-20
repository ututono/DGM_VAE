from core.agent import AbstractAgent
import torch

from torch.utils.data import DataLoader
from core.optimizer import Optimizer
from core.loss_function import LossFunction
from core.utils.metrics import Metrics
from typing import List


class VariationalAutoEncoder(AbstractAgent):
    def __init__(self, model, device, metrics):
        ...



    def _perform_training_epoch(self, epoch, train_dataloader: DataLoader,
                                optimizer: Optimizer,
                                loss_fn: LossFunction):
        ...
            

    def _perform_evaluation_epoch(self, epoch, eval_dataloader: DataLoader, loss_fn: LossFunction):
        ...


    def test(self, test_data):
        return


    def predict(self, x: torch.Tensor) -> torch.Tensor:
        ...