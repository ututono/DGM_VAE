import os
import random
import torch
import numpy as np

import rootutils


def set_random_seed(seed: int = 42):
    """Set the seed for determinism"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def root_path():
    root = str(os.getenv("PROJECT_ROOT", None))
    if root is None:
        root = str(rootutils.find_root(search_from=__file__, indicator=".project-root"))
    return root

