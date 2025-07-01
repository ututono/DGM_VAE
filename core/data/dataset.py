import torch
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List
from core.data.backend import NumpyBackend, TorchBackend
import torch.nn.functional as F

from torchvision import transforms

import medmnist
from medmnist import INFO

from core.configs.values import DataSplitType


class Dataset():
    def __init__(self, sources, targets = None, device: str = 'cpu', backend: str = 'numpy', load_data: bool = False,
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
    

    def __repr__(self):
        return f"Number of samples: {self._sources.size(0)}"


def load_medmnist_data(
        dataset_name: str ='pathmnist',
        image_size: int = 28,
        download=True,
        custom_transform:Optional[List] = None,
        as_rgb: bool = False
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, dict]:
    """
    Load MedMNIST dataset
    @:param dataset_name: Name of the MedMNIST dataset to load
    @:param data_flag: Size flag for the dataset (e.g., '28' for 28x28 images)
    @:param download: Whether to download the dataset if not already present
    @:return: Tuple of train_loader, val_loader, test_loader, info

    """
    def _load_dataset(data_class, split, transform, download, image_size, as_rgb):
        return data_class(
            split=split,
            transform=transform,
            download=download,
            as_rgb=as_rgb,
            size=image_size
        )

    info = INFO[dataset_name.lower()]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    # Define transforms
    if custom_transform is not None:
        data_transform = transforms.Compose(custom_transform)
    else:
        # Default transforms for MedMNIST datasets according to an [example](https://colab.research.google.com/github/MedMNIST/MedMNIST/blob/main/examples/getting_started.ipynb#scrollTo=uJDrVvTmfUyE&line=7&uniqifier=1) from the medmnist repository
        data_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    # # Load datasets
    train_dataset = _load_dataset(
        DataClass,
        split=DataSplitType.TRAIN,
        transform=data_transform,
        download=download,
        image_size=image_size,
        as_rgb=as_rgb
    )

    val_dataset = _load_dataset(
        DataClass,
        split=DataSplitType.VALIDATION,
        transform=data_transform,
        download=download,
        image_size=image_size,
        as_rgb=as_rgb
    )

    test_dataset = _load_dataset(
        DataClass,
        split=DataSplitType.TEST,
        transform=data_transform,
        download=download,
        image_size=image_size,
        as_rgb=as_rgb
    )

    return train_dataset, val_dataset, test_dataset, info
