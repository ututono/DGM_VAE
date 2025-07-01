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
            # 'test': test_metrics
        }
        super().__init__(model=model, device=device, metrics=metrics)

    def _perform_training_epoch(self, epoch, train_dataloader: DataLoader,
                                optimizer: Optimizer,
                                loss_fn: LossFunction):

        epoch_losses = list()
        total_samples = 0

        for x, _ in train_dataloader:
            x: torch.Tensor = x.to(self._device)

            x_hat, mu, logvar = self._model(x)
            loss: torch.Tensor = loss_fn(x_hat, mu, logvar, x)
            epoch_losses.append(loss.item())
            total_samples += x.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss: float = sum(epoch_losses) / total_samples
        self._metrics['train'].update(epoch=epoch, batch_loss=epoch_loss)

    def _perform_evaluation_epoch(self, epoch, eval_dataloader: DataLoader, loss_fn: LossFunction):

        epoch_losses_val = list()
        total_samples = 0

        with torch.no_grad():
            for x, _ in eval_dataloader:  # x: (images, labels)
                x_val: torch.Tensor = x.to(self._device)
                xhat_val, mu, logvar = self._model(x_val)

                loss: torch.Tensor = loss_fn(xhat_val, mu, logvar, x_val)
                epoch_losses_val.append(loss.item())
                total_samples += x_val.size(0)

        epoch_loss_validation: float = sum(epoch_losses_val) / total_samples
        self._metrics['validation'].update(epoch=epoch, batch_loss=epoch_loss_validation)

        # self._metrics['validation'].update(epoch=epoch, batch_loss=epoch_loss_validation)

    def test(self, test_data):
        """Test the model on test data"""
        self._model.eval()
        test_losses = []
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(test_data):
                x_test = x.to(self._device)

                x_hat_test, mu, logvar = self._model(x_test)

                loss = torch.nn.functional.binary_cross_entropy(x_hat_test, x_test, reduction='sum')
                test_losses.append(loss.item())
                total_samples += x_test.size(0)

        average_test_loss = sum(test_losses) / total_samples
        return {'test_loss(recon_loss)': average_test_loss}

    def predict(self, num_samples: int = 1) -> torch.Tensor:
        self._model.eval()
        latent_dim = self._model.latent_dim if hasattr(self._model, 'latent_dim') else 128
        with torch.no_grad():
            # Sample from standard normal distribution
            z = torch.randn(num_samples, latent_dim).to(self._device)

            # Generate samples using the decoder
            if hasattr(self._model, 'decode'):
                generated_samples = self._model.decode(z)
            else:
                raise NotImplementedError("Model must have a 'decode' method for generation")

            return generated_samples

    def load_parameters(self, path):
        """Load model parameters from a given path."""
        self._model.load_state_dict(torch.load(path, map_location=self._device))
        self._model.to(self._device)

    def save_parameters(self, path):
        """Save model parameters to a given path."""
        torch.save(self._model.state_dict(), path)
