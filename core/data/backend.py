from abc import ABC, abstractmethod

from typing import Tuple
import PIL.Image
import torch
import numpy as np
from PIL.Image import Image

class Backend(ABC):

    def __init__(self, device):
        self._device = device

    @abstractmethod
    def load(src_path: str, target_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()
    

    @abstractmethod
    def to_array(data):
        raise NotImplementedError()
    

    @abstractmethod
    def to_tensor(data):
        raise NotImplementedError()
    


class NumpyBackend(Backend):

    def __init__(self, device):
        super().__init__(device=device)


    def load(self, src_path: str, target_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        source = np.load(src_path)
        targets = np.load(target_path)
        return torch.from_numpy(source), torch.from_numpy(targets)
    

    def to_array(self, data: np.ndarray | torch.Tensor) -> np.ndarray:
        return data.numpy() if isinstance(data, torch.Tensor) else data
    

    def to_tensor(self, data: np.ndarray | torch.Tensor | Image) -> torch.Tensor:
        return torch.tensor(data) if not isinstance(data, torch.Tensor) else data
    

class TorchBackend(Backend):
    
    def __init__(self, device):
        super().__init__(device=device)


    def load(self, src_path: str, target_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        source = np.load(src_path)
        targets = np.load(target_path)
        return torch.from_numpy(source), torch.from_numpy(targets)
    

    def to_array(self, data: np.ndarray | torch.Tensor) -> np.ndarray:
        return data.numpy() if isinstance(data, torch.Tensor) else data
    

    def to_tensor(self, data: np.ndarray | torch.Tensor | Image) -> torch.Tensor:
        return torch.tensor(data) if not isinstance(data, torch.Tensor) else data