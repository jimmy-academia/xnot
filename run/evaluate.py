#!/usr/bin/env python3
"""Ranking evaluation functions."""

import inspect
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from data.loader import format_ranking_query
from utils.parsing import parse_indices
from utils.usage import get_usage_tracker

from .shuffle import apply_shuffle, unmap_predictions


class ContextLengthExceeded(Exception):
    """Raised when LLM context length is exceeded."""
    pass


def extract_hits_at(stats: dict, k: int = 5) -> dict:
    """Extract hits_at accuracies from stats dict.

    Args:
        stats: Stats dict with hits_at data
        k: Maximum K value to extract

    Returns:
        Dict with hits_at_1, hits_at_2, ..., hits_at_k keys
    """
    hits_at = stats.get("hits_at", {})
    result = {}
    for i in range(1, k + 1):
        acc = hits_at.get(i, hits_at.get(str(i), {})).get("accuracy", 0)
        result[f"hits_at_{i}"] = acc
    return result


def compute_multi_k_stats(results: list[dict], k: int) -> dict:
    """Compute Hits@1 through Hits@k from results.

    Args:
        results: List of result dicts with 'gold_idx' and 'pred_indices'
        k: Maximum k value to compute

    Returns:
        Dict with total, k, and hits_at dict for each level
    """
    total = len(results)
    hits_at = {}
    for j in range(1, k + 1):
        hits = sum(1 for r in results
                   if (r["gold_idx"] + 1) in r["pred_indices"][:j])
        hits_at[j] = {"hits": hits, "accuracy": hits / total if total else 0}
    return {
        "total": total,
        "k": k,
        "hits_at": hits_at,
    }


def evaluate_ranking_single(method, items: list, mode: str, shuffle: str,
                            context: str, k: int, req: dict,
                            groundtruth: dict) -> dict | None:
    """Evaluate a single request (thread-safe helper).

    Args:
        method: LLM method instance
        items: All items (will be shuffled per request)
        mode: "string" or "dict" for formatting
        shuffle: Shuffle strategy ("none", "middle", "random")
        context: Request context/text
        k: Number of top predictions
        req: Request dict with 'id'
        groundtruth: {request_id: {"gold_restaurant": str, "gold_idx": int}}

    Returns:
        Result dict or None if no ground truth

    Raises:
        ContextLengthExceeded: If the context length limit is exceeded
    """
    req_id = req["id"]
    gt = groundtruth.get(req_id)
    if not gt:
        return None

    gold_idx = gt["gold_idx"]

    # Apply shuffle based on gold position
    shuffled_items, mapping, shuffled_gold_pos = apply_shuffle(items, gold_idx, shuffle)

    # Format query with shuffled items
    query, item_count = format_ranking_query(shuffled_items, mode)

    # Track per-request token usage
    tracker = get_usage_tracker()
    start_idx = len(tracker.get_records())

    response = None
    shuffled_preds = []
    try:
        # Only pass request_id if method accepts it (e.g., ANoT)
        sig = inspect.signature(method.evaluate_ranking)
        if 'request_id' in sig.parameters:
            response = method.evaluate_ranking(query, context, k=k, request_id=req_id)
        else:
            response = method.evaluate_ranking(query, context, k=k)
        shuffled_preds = parse_indices(response, item_count, k)

    except Exception as e:
        error_str = str(e)
        if "context_length_exceeded" in error_str or "too many tokens" in error_str.lower():
            raise ContextLengthExceeded(error_str)
        shuffled_preds = []

    # Compute usage for this request
    request_records = tracker.get_records()[start_idx:]
    request_prompt_tokens = sum(r['prompt_tokens'] for r in request_records)
    request_completion_tokens = sum(r['completion_tokens'] for r in request_records)
    request_tokens = request_prompt_tokens + request_completion_tokens
    request_cost = sum(r['cost_usd'] for r in request_records)
    request_latency = sum(r['latency_ms'] for r in request_records)

    # Map predictions back to original indices
    pred_indices = unmap_predictions(shuffled_preds, mapping)

    return {
        "request_id": req_id,
        "pred_indices": pred_indices,
        "shuffled_preds": shuffled_preds,
        "gold_idx": gold_idx,
        "shuffled_gold_pos": shuffled_gold_pos + 1,
        "gold_restaurant": gt["gold_restaurant"],
        "prompt_tokens": request_prompt_tokens,
        "completion_tokens": request_completion_tokens,
        "tokens": request_tokens,
        "cost_usd": request_cost,
        "latency_ms": request_latency,
    }


def _run_with_progress(generator, has_rich_display: bool, description: str, total: int):
    """Run a generator with optional progress display."""
    if has_rich_display:
        for _ in generator:
            pass
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task(description, total=total)
            for _ in generator:
                progress.update(task, advance=1)


def evaluate_ranking(items: list[dict], method: Callable, requests: list[dict],
                     groundtruth: dict, mode: str = "string", k: int = 5,
                     shuffle: str = "middle", parallel: bool = True,
                     max_workers: int = 40) -> dict:
    """Evaluate using ranking (Hits@K accuracy).

    Args:
        items: All items (restaurants) to rank
        method: LLM method that returns ranking via evaluate_ranking()
        requests: List of requests
        groundtruth: {request_id: {"gold_restaurant": str, "gold_idx": int}}
        mode: "string" or "dict" for formatting
        k: Number of top predictions to check (default 5 for Hits@5)
        shuffle: Shuffle strategy ("none", "middle", "random") - default "middle"
        parallel: Whether to use parallel execution (default True)
        max_workers: Maximum number of worker threads (default 40)

    Returns:
        Dict with results and accuracy stats
    """
    # Start rich Live display if method supports it (e.g., ANoT)
    has_rich_display = hasattr(method, 'start_display') and hasattr(method, 'stop_display')
    if has_rich_display:
        title = f"ANoT: {len(items)} candidates, k={k}"
        method.start_display(title=title, total=len(requests), requests=requests)

    try:
        return _evaluate_ranking_inner(
            items, method, requests, groundtruth, mode, k, shuffle, parallel, max_workers, has_rich_display
        )
    finally:
        if has_rich_display:
            method.stop_display()


def _evaluate_ranking_inner(items, method, requests, groundtruth, mode, k, shuffle, parallel, max_workers, has_rich_display):
    """Inner implementation of evaluate_ranking (wrapped by display context)."""
    req_ids = [r["id"] for r in requests]
    context_exceeded = False
    results = []

    if parallel:
        def run_eval():
            nonlocal context_exceeded
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        evaluate_ranking_single,
                        method, items, mode, shuffle,
                        req.get("context") or req.get("text", ""),
                        k, req, groundtruth
                    ): req
                    for req in requests
                }

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except ContextLengthExceeded:
                        context_exceeded = True
                        for f in futures:
                            f.cancel()
                        break
                    yield 1

        description = f"Ranking evaluation (parallel, {max_workers} workers)..."
    else:
        def run_eval():
            nonlocal context_exceeded
            for req in requests:
                context = req.get("context") or req.get("text", "")
                try:
                    result = evaluate_ranking_single(
                        method, items, mode, shuffle, context, k, req, groundtruth
                    )
                    if result:
                        results.append(result)
                except ContextLengthExceeded:
                    context_exceeded = True
                    return
                yield 1

        description = "Ranking evaluation (sequential)..."

    _run_with_progress(run_eval(), has_rich_display, description, len(requests))

    stats = compute_multi_k_stats(results, k)

    return {
        "results": results,
        "req_ids": req_ids,
        "stats": stats,
        "context_exceeded": context_exceeded,
    }
