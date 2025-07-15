from os import PathLike

from core.agent import AbstractAgent
from core.optimizer import Optimizer
from core.loss_function import LossFunction
from core.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

from typing import Tuple, Any, Optional, Dict
from pathlib import Path


class Core():
    def __init__(
            self,
            agent: AbstractAgent,
            optimizer: Optional[Optimizer],
            loss_function: LossFunction,
            num_workers: int = 0,  # Number of workers for DataLoader
    ):
        self._agent = agent
        self._optimizer = optimizer
        self._critetion = loss_function
        self._num_workers = num_workers

    def train(
            self,
            training_data,
            evaluation_data=None,
            batch_size=64, epochs=100,
            quiet=False,
            collate_fn=None,
            train_sampler=None,
            val_sampler=None
    ):
        training_dataloader = DataLoader(
            training_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            collate_fn=collate_fn,
            sampler=train_sampler
        )
        evaluation_dataloader = DataLoader(
            evaluation_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            collate_fn=collate_fn,
            sampler=val_sampler
        ) if evaluation_data else None
        return self._agent.train(training_dataloader, evaluation_dataloader,
                                 self._optimizer, self._critetion,
                                 epochs=epochs, quiet=quiet)

    def test(self, data, batch_size=1, collate_fn=None, sampler=None):
        test_dataloader = DataLoader(
            data,
            batch_size=batch_size,
            num_workers=self._num_workers,
            collate_fn=None, 
            sampler=None
        )
        return self._agent.test(test_dataloader)

    def build_dataset(self, *data,
                      backend_type: str = "numpy", one_hot: bool = False,
                      num_classes: bool = 0, device: str = "cpu"):
        assert len(data) > 0, "No data given for building a dataset"
        load_data = False
        if isinstance(data[0], (str, Path)):
            load_data = True

        return Dataset(sources=data[0], targets=data[1], device=device, backend=backend_type,
                       load_data=load_data, one_hot=one_hot, num_classes=num_classes)

    @staticmethod
    def split_data_train_test(src, target, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(src, target, test_size=test_size)

        return X_train, y_train, X_test, y_test

    def generate_visual_report(self, artifacts_dir: PathLike = "outputs", dataset_info: Optional[Dict[str, Any]] = None,
                               data=None):
        if not isinstance(artifacts_dir, Path):
            artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        agent = self._agent
        # Generate a report based on the model's output
        if agent.is_conditional_training:
            from core.visualization.visualize_model_output import generate_cvae_report

            data_loader = DataLoader(data, batch_size=64, shuffle=True, num_workers=self._num_workers) if data else None
            generate_cvae_report(agent, artifacts_dir=artifacts_dir, dataset_info=dataset_info, data_loader=data_loader)
