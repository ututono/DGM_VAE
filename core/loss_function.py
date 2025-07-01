import torch.nn as nn
import torch
import torch.nn.functional as F


class LossFunction(nn.Module):
    def __init__(self, loss_function: nn.Module | str, device: str):
        super().__init__()
        self._device = device
        if isinstance(loss_function, str):
            self._loss_function: nn.Module = getattr(nn, loss_function)()
        else:
            self._loss_function: nn.Module = loss_function

        self._loss_function.to(device=device)


    def __repr__(self):
        return f"Loss Function: {self._loss_function}"
    

    def _get_targets(self, targets: torch.Tensor) -> torch.Tensor:
        if isinstance(self._loss_function, (nn.CrossEntropyLoss, nn.NLLLoss)):
            return torch.max(targets, 1)[1].to(self._device)
        return targets.to(self._device)


    def forward(self, x_hat, mu, logvar, x) -> torch.Tensor:
        return self._loss_function(x_hat, mu, logvar, x)


class VAELoss(nn.Module):
    def __init__(self, beta=1.0):
        """
        Variational Autoencoder Loss Function.
        :param beta: Weighting factor for the KL divergence term.
        """
        super().__init__()
        self.beta = beta

    def forward(self, x_hat, mu, logvar, x):
        bce = F.binary_cross_entropy(x_hat, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce + self.beta * kld
