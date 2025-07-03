from os import PathLike

from core.agent import AbstractAgent
import torch

from torch.utils.data import DataLoader
from core.optimizer import Optimizer
from core.loss_function import LossFunction
from core.utils.metrics import Metrics
from typing import List, Dict, Any


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
        self.records_manager = None  # Will be set in save_checkpoint/load_checkpoint methods
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

        n_samples = 8
        comparison = None

        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(test_data):
                x_test = x.to(self._device)

                x_hat_test, mu, logvar = self._model(x_test)

                if batch_idx == 0:
                    comparison = torch.cat([x_test[:n_samples], x_hat_test[:n_samples]])

                loss = torch.nn.functional.binary_cross_entropy(x_hat_test, x_test, reduction='sum')
                test_losses.append(loss.item())
                total_samples += x_test.size(0)

        average_test_loss = sum(test_losses) / total_samples
        results = {
            'test_loss(recon_loss)': average_test_loss,
            'comparison': comparison
        }
        return results

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

    def reconstruct(self, x):
        """Reconstruct input images using the VAE model."""
        self._model.eval()
        x = x.to(self._device)
        with torch.no_grad():
            x_hat, mu, logvar = self._model(x)
            return x_hat, mu, logvar

    def save_checkpoint(self, path: PathLike, args: Dict[str, Any] = None):
        """Save the model and its training records to a checkpoint file."""
        from core.utils.training_records import save_checkpoint

        current_metrics = self.get_current_metrics_summary()

        self.records_manager = save_checkpoint(
            model=self._model,
            save_dir=path,
            args=args,
            metrics=current_metrics,
            records_manager=self.records_manager
        )

        return self.records_manager

    def load_checkpoint(self, checkpoint_path: PathLike, current_args, force_continue: bool = False):
        """Load the model and its training records from a checkpoint file."""
        from core.utils.training_records import load_checkpoint

        self.records_manager = load_checkpoint(
            model=self._model,
            load_dir=checkpoint_path,
            current_args=current_args,
            force_continue=force_continue
        )

        self._model.to(self._device)

        return self.records_manager

    def get_current_metrics_summary(self):
        """Get summary of current training metrics for final report."""
        train_losses = self._metrics['train']['loss']
        val_losses = self._metrics['validation']['loss']

        summary = {
            'epochs_completed': len(train_losses),
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'best_val_loss': min(val_losses) if val_losses else None,
            'best_val_epoch': val_losses.index(min(val_losses)) + 1 if val_losses else None
        }

        return summary