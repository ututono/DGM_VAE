import errno
import os
import random
from pathlib import Path
from typing import Optional

import torch
import numpy as np


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