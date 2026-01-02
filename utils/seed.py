"""Seed and configuration utilities."""

import os
import random


def set_seeds(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
