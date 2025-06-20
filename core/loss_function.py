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
        return targets


    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        labels = self._get_targets(target)
        return self._loss_function(prediction, labels)
