"""Seed and configuration utilities."""

import os
import random


def set_seeds(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_key():
    """Load OpenAI API key from file."""
    with open("../.openaiapi", "r") as f:
        return f.read().strip()
