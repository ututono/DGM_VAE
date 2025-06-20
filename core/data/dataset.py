import torch
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple
from core.data.backend import NumpyBackend, TorchBackend
import torch.nn.functional as F


class Dataset():
    def __init__(self, sources, targets, device: str, backend: str = 'numpy', load_data: bool = False,
                 one_hot: bool = False, num_classes: int = -1):
        if backend == "numpy":
            self._backend = NumpyBackend(device=device)
        else:
            self._backend = TorchBackend(device=device)

        self._sources, self._targets = self._get_data(sources, targets, load_data)
        self._length = self._sources.size(0)

        if one_hot:
            assert num_classes > 1, "num_classes must be bigger that one to apply one hot encoding"
            self._targets = F.one_hot(self._targets, num_classes=num_classes).float()


    def _get_data(self, src, target, load_data: bool = False):
        if load_data:
            sources, targets = self._backend.load(src, target)
            return sources.float(), targets.float()
        
        return self._backend.to_tensor(src).float(), self._backend.to_tensor(target).float()


    def __len__(self):
        return self._length
    

    def __getitem__(self, index):
        return self._sources[index], self._targets[index]
    