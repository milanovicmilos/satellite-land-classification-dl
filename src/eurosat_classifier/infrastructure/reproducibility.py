"""Reproducibility utilities for deterministic training behavior."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch backends."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Prefer deterministic kernels to reduce run-to-run variance across machines.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
