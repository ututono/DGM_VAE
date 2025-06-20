from abc import abstractmethod, ABC
import torch.nn as nn
from torch.utils.data import DataLoader
from core.optimizer import Optimizer
from core.loss_function import LossFunction
from core.utils.metrics import Metrics

import torch
from typing import List, Dict
from tqdm import tqdm

class AbstractAgent(ABC):
    def __init__(self, model: nn.Module, device: str, metrics: Dict[str, Metrics]):
        self._model: nn.Module = model.to(device=device)
        self._device = device
        self._metrics: Dict[str, Metrics] = metrics


    @abstractmethod
    def _perform_training_epoch(self, epoch, train_dataloader, optimizer, loss_fn):
        raise NotImplementedError
    

    @abstractmethod
    def _perform_evaluation_epoch(self, epoch, eval_dataloader: DataLoader, loss_fn: LossFunction):
        raise NotImplementedError

    
    def train(self, training_data: DataLoader, evaluation_data: DataLoader,
              optimizer: Optimizer, loss_fn: LossFunction,
              start_epoch: int = 0, epochs: int = 100):
        assert start_epoch < epochs, f"start_epoch (given {start_epoch}) must be less than epochs (given {epochs})"

        with tqdm(total=epochs, leave=True, position=0) as pbar:
            for epoch in range(start_epoch, epochs):
                self._model.train()
                self._perform_training_epoch(epoch=epoch, 
                                            train_dataloader=training_data,
                                            optimizer=optimizer,
                                            loss_fn=loss_fn)

                if evaluation_data is not None:
                    self._model.eval()
                    self._perform_evaluation_epoch(epoch, evaluation_data, loss_fn)
                    
                pbar.update(1)
                postfix_metrics = {'train_loss': self._metrics['train']['loss'][-1], 'train_acc': self._metrics['train']['accuracy'][-1],
                                   'val_loss': self._metrics['validation']['loss'][-1], 'val_acc': self._metrics['validation']['accuracy'][-1]
                                   }
                pbar.set_postfix(**postfix_metrics)

        return self._metrics['train'], self._metrics['validation']
    

    @abstractmethod
    def test(self, test_data: DataLoader):
        raise NotImplementedError
    

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError
    

    def load_parameters(self, path):
        pass


    def save_parameters(self, path):
        pass

    
    def get_parameters(self):
        return self._model.parameters()