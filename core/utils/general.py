import errno
import logging
import os
import random
from pathlib import Path
from typing import Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)

def set_random_seed(seed: int = 42):
    """Set the seed for determinism"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def root_path():
    from dotenv import load_dotenv
    import rootutils

    load_dotenv()  # Take environment variables from .env.

    rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

    root = str(os.getenv("PROJECT_ROOT", None))
    if root is None:
        root = str(rootutils.find_root(search_from=__file__, indicator=".project-root"))
    return root


def check_and_mkdir(path: Path | str, parents=True, exist_ok=True, non_exist_action="mkdir", suffix_is_file=False) -> \
        Optional[bool]:
    """
    Check if the folder exists, if not, create it.
    :param path: The path to check.
    :param parents: Whether to create parent directories if needed.
    :param exist_ok: Whether it's okay if the folder already exists.
    :param non_exist_action: Action to take if the folder doesn't exist. Options: "mkdir", "nothing".
    :param suffix_is_file: Used to determine how to handle a path without file extension. If true, the path is treated as
    a directory, otherwise the path is treated as a file without file extension.
    :return: True if the folder exists or was created; False if it does not exist and action is "nothing".
    """

    def is_dir(path_: Path):
        """A path is considered as a directory if it has no suffix."""
        if not path_.is_dir():
            if path_.suffix == "" and suffix_is_file:
                return True
            else:
                return False
        return True

    path = Path(path)
    folder = path if is_dir(path) else path.parent
    if not folder.exists():
        if non_exist_action == "mkdir":
            folder.mkdir(parents=parents, exist_ok=exist_ok)
            return True
        elif non_exist_action == "nothing":
            return False
        else:
            raise FileNotFoundError(f"Folder {folder} does not exist.")
    else:
        # logger.debug(f"Folder {folder} exists.")
        return True


def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def apply_smoke_test_settings(args, train_ds, val_ds, test_ds):
    """
    Apply smoke test settings to reduce resource usage for CI/CD.
    """
    if args.smoke_test:
        logger.info("Smoke test mode enabled - reducing dataset size and training parameters")

        # Limit dataset sizes
        train_subset_size = min(10, len(train_ds))
        val_subset_size = min(2, len(val_ds))
        test_subset_size = min(1, len(test_ds))

        # Create subsets
        train_indices = torch.randperm(len(train_ds))[:train_subset_size]
        val_indices = torch.randperm(len(val_ds))[:val_subset_size]
        test_indices = torch.randperm(len(test_ds))[:test_subset_size]

        train_ds = torch.utils.data.Subset(train_ds, train_indices)
        val_ds = torch.utils.data.Subset(val_ds, val_indices)
        test_ds = torch.utils.data.Subset(test_ds, test_indices)

        # Override training parameters for speed
        args.epochs = 1
        args.batch_size = 2
        args.latent_dim = min(args.latent_dim, 32)  # Reduce latent dimension
        args.learning_rate = 0.01  # Slightly higher for faster convergence

        logger.info(f"Smoke test settings applied:\n"
                    f"  - Train samples: {len(train_ds)}\n"
                    f"  - Val samples: {len(val_ds)}\n"
                    f"  - Test samples: {len(test_ds)}\n"
                    f"  - Epochs: {args.epochs}\n"
                    f"  - Batch size: {args.batch_size}\n"
                    f"  - Latent dim: {args.latent_dim}")

    return train_ds, val_ds, test_ds
