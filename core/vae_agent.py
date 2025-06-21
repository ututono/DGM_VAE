from core.agent import AbstractAgent
import torch

from torch.utils.data import DataLoader
from core.optimizer import Optimizer
from core.loss_function import LossFunction
from core.utils.metrics import Metrics
from typing import List


class VariationalAutoEncoder(AbstractAgent):
    def __init__(self, model, device):
        train_metrics = Metrics(["loss"])
        val_metrics = Metrics(["loss"])
        test_metrics = Metrics(["loss"])

        metrics = {
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics
        }
        super().__init__(model=model, device=device, metrics=metrics)



    def _perform_training_epoch(self, epoch, train_dataloader: DataLoader,
                                optimizer: Optimizer,
                                loss_fn: LossFunction):
        
        epoch_losses = list()
        total_samples = 0

        for x, y in train_dataloader:
            x: torch.Tensor = x.to(self._device)
            y: torch.Tensor = y.unsqueeze(-1).to(self._device)

            yhat: torch.Tensor = self._model(x)
            loss: torch.Tensor = loss_fn(yhat, y)
            epoch_losses.append(loss.item())
            total_samples += y.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        epoch_loss: float = sum(epoch_losses) / total_samples
        self._metrics['train'].update(epoch=epoch, batch_loss=epoch_loss)

            

    def _perform_evaluation_epoch(self, epoch, eval_dataloader: DataLoader, loss_fn: LossFunction):
        
        epoch_losses_val = list()
        total_samples = 0
        
        with torch.no_grad():
            for x, y in eval_dataloader:
                x_val: torch.Tensor = x.to(self._device)
                y_val: torch.Tensor = y.unsqueeze(-1).to(self._device)
                yhat_val = self._model(x_val)

                loss: torch.Tensor = loss_fn(yhat_val, y_val)
                epoch_losses_val.append(loss.item())
                total_samples += y_val.size(0)
                epoch_loss_validation: float = sum(epoch_losses_val) / total_samples

            self._metrics['validation'].update(epoch=epoch, batch_loss=epoch_loss_validation)


    def test(self, test_data):
        return


    def predict(self, x: torch.Tensor) -> torch.Tensor:
        ...