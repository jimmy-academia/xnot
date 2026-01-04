#!/usr/bin/env python3
"""Run package - evaluation orchestration and scaling experiments."""

from .shuffle import shuffle_gold_to_middle, apply_shuffle, unmap_predictions
from .io import (
    load_existing_results, load_usage, save_usage,
    extract_usage_from_results, load_failed_scales, save_scaling_summary
)
from .evaluate import (
    ContextLengthExceeded, extract_hits_at, compute_multi_k_stats,
    evaluate_ranking_single, evaluate_ranking
)
from .orchestrate import run_evaluation_loop, save_final_config, run_single
from .scaling import SCALE_POINTS, run_scaling_experiment


__all__ = [
    # Shuffle
    "shuffle_gold_to_middle",
    "apply_shuffle",
    "unmap_predictions",
    # I/O
    "load_existing_results",
    "load_usage",
    "save_usage",
    "extract_usage_from_results",
    "load_failed_scales",
    "save_scaling_summary",
    # Evaluate
    "ContextLengthExceeded",
    "extract_hits_at",
    "compute_multi_k_stats",
    "evaluate_ranking_single",
    "evaluate_ranking",
    # Orchestrate
    "run_evaluation_loop",
    "save_final_config",
    "run_single",
    # Scaling
    "SCALE_POINTS",
    "run_scaling_experiment",
]
