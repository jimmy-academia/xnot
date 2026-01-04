#!/usr/bin/env python3
"""I/O utilities for result and usage loading/saving."""

import json
from datetime import datetime
from pathlib import Path


def load_existing_results(run_dir: Path, n_candidates: int) -> dict:
    """Load existing results for a candidate count.

    Returns:
        Dict of {request_id: result_dict}
    """
    results_path = run_dir / f"results_{n_candidates}.jsonl"
    if not results_path.exists():
        return {}

    existing = {}
    for line in results_path.read_text().strip().split("\n"):
        if line:
            r = json.loads(line)
            existing[r["request_id"]] = r
    return existing


def load_usage(run_dir: Path) -> dict:
    """Load existing usage records from usage.jsonl.

    Returns:
        Dict keyed by (request_id, n_candidates) -> usage record
    """
    usage_path = run_dir / "usage.jsonl"
    if not usage_path.exists():
        return {}

    existing = {}
    for line in usage_path.read_text().strip().split("\n"):
        if line:
            r = json.loads(line)
            key = (r["request_id"], r["n_candidates"])
            existing[key] = r
    return existing


def save_usage(run_dir: Path, usage_dict: dict):
    """Save usage records to usage.jsonl.

    Args:
        run_dir: Run directory path
        usage_dict: Dict keyed by (request_id, n_candidates) -> usage record
    """
    usage_path = run_dir / "usage.jsonl"
    sorted_records = sorted(usage_dict.values(), key=lambda x: (x["n_candidates"], x["request_id"]))
    with open(usage_path, "w") as f:
        for r in sorted_records:
            f.write(json.dumps(r) + "\n")


def extract_usage_from_results(results: list, n_candidates: int) -> dict:
    """Extract per-request usage records from results.

    Args:
        results: List of result dicts from evaluation
        n_candidates: Number of candidates for this run

    Returns:
        Dict keyed by (request_id, n_candidates) -> usage record
    """
    timestamp = datetime.now().isoformat()

    usage_dict = {}
    for r in results:
        key = (r["request_id"], n_candidates)
        usage_dict[key] = {
            "request_id": r["request_id"],
            "n_candidates": n_candidates,
            "prompt_tokens": r.get("prompt_tokens", 0),
            "completion_tokens": r.get("completion_tokens", 0),
            "tokens": r.get("tokens", 0),
            "cost_usd": r.get("cost_usd", 0),
            "latency_ms": r.get("latency_ms", 0),
            "timestamp": timestamp,
        }
    return usage_dict


def load_failed_scales(run_dir: Path) -> set:
    """Load scales that already hit context limit.

    Returns:
        Set of candidate counts that failed (context_exceeded or skipped)
    """
    summary_path = run_dir / "scaling_summary.json"
    if not summary_path.exists():
        return set()

    summary = json.loads(summary_path.read_text())
    failed = set()
    for r in summary.get("results", []):
        if r.get("status") in ("context_exceeded", "skipped"):
            failed.add(r["candidates"])
    return failed


def save_scaling_summary(run_dir: Path, results_table: list, k: int, scale_points: list):
    """Save scaling experiment summary to JSON.

    Args:
        run_dir: Run directory path
        results_table: List of {candidates, requests, hits_at_1, hits_at_5, status}
        k: K value used for evaluation
        scale_points: List of scale points used
    """
    summary = {
        "scale_points": scale_points,
        "k": k,
        "results": results_table,
    }
    summary_path = run_dir / "scaling_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nScaling summary saved to {summary_path}")
