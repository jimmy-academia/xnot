#!/usr/bin/env python3
"""Ranking evaluation functions."""

import inspect
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from data.loader import format_ranking_query, format_ranking_query_packed
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


def _apply_attack_if_needed(
    items: list,
    attack_config: dict | None,
    gold_id: str,
    req_id: str,
) -> list:
    """Apply attack to items if configured, protecting gold.

    Args:
        items: List of items (restaurants)
        attack_config: Attack configuration dict or None
        gold_id: ID of gold restaurant to protect
        req_id: Request ID for reproducible seeding

    Returns:
        Modified items list (or original if no attack)
    """
    if not attack_config:
        return items
    if attack_config.get("attack", "none") in ("none", "clean", None, ""):
        return items

    from attack import apply_attack_for_request

    # Use request-specific seed for reproducibility
    base_seed = attack_config.get("seed")
    request_seed = hash(req_id) % (2**31) if base_seed is None else base_seed + hash(req_id) % 1000

    return apply_attack_for_request(items, attack_config, gold_id, request_seed)


def _format_context(
    items: list,
    mode: str,
    token_budget: int | None,
    model: str | None,
) -> tuple:
    """Format items as context for the model.

    Args:
        items: List of items (already shuffled)
        mode: "string" or "dict" for formatting
        token_budget: Optional token budget for truncation
        model: Model name for tokenizer

    Returns:
        (context, item_count, coverage_stats) - coverage_stats is None if not truncated
    """
    if token_budget and mode == "string":
        context, item_count, coverage_stats = format_ranking_query_packed(
            items, token_budget, model or "gpt-4o"
        )
        return context, item_count, coverage_stats

    context, item_count = format_ranking_query(items, mode)
    return context, item_count, None


def _invoke_method(method, query: str, context, k: int, req_id: str) -> str:
    """Invoke the method with appropriate signature.

    Args:
        method: LLM method instance
        query: User request text
        context: Formatted context (string or dict)
        k: Number of predictions
        req_id: Request ID (passed if method accepts it)

    Returns:
        Method response string
    """
    sig = inspect.signature(method.evaluate_ranking)
    if 'request_id' in sig.parameters:
        return method.evaluate_ranking(query, context, k=k, request_id=req_id)
    return method.evaluate_ranking(query, context, k=k)


def _build_result(
    req_id: str,
    pred_indices: list,
    shuffled_preds: list,
    gold_idx: int,
    shuffled_gold_pos: int,
    gt: dict,
    usage_records: list,
    coverage_stats: dict | None,
) -> dict:
    """Assemble the result dictionary.

    Args:
        req_id: Request ID
        pred_indices: Predictions mapped to original indices
        shuffled_preds: Predictions in shuffled order
        gold_idx: Gold index in original order
        shuffled_gold_pos: Gold position in shuffled order
        gt: Ground truth dict
        usage_records: List of usage records for this request
        coverage_stats: Coverage stats or None

    Returns:
        Result dict
    """
    prompt_tokens = sum(r['prompt_tokens'] for r in usage_records)
    completion_tokens = sum(r['completion_tokens'] for r in usage_records)

    result = {
        "request_id": req_id,
        "pred_indices": pred_indices,
        "shuffled_preds": shuffled_preds,
        "gold_idx": gold_idx,
        "shuffled_gold_pos": shuffled_gold_pos + 1,
        "gold_restaurant": gt["gold_restaurant"],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens": prompt_tokens + completion_tokens,
        "cost_usd": sum(r['cost_usd'] for r in usage_records),
        "latency_ms": sum(r['latency_ms'] for r in usage_records),
    }

    if coverage_stats:
        result["coverage"] = coverage_stats

    return result


def evaluate_ranking_single(method, items: list, mode: str, shuffle: str,
                            query: str, k: int, req: dict,
                            groundtruth: dict, attack_config: dict = None,
                            token_budget: int = None, model: str = None) -> dict | None:
    """Evaluate a single request (thread-safe helper).

    Args:
        method: LLM method instance
        items: All items (will be shuffled per request)
        mode: "string" or "dict" for formatting
        shuffle: Shuffle strategy ("none", "middle", "random")
        query: User request text (what user is searching for)
        k: Number of top predictions
        req: Request dict with 'id'
        groundtruth: {request_id: {"gold_restaurant": str, "gold_idx": int}}
        attack_config: Optional attack configuration for per-request attacks
        token_budget: Optional token budget for pack-to-budget truncation (string mode only)
        model: Model name for tokenizer (required if token_budget is set)

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
    gold_id = gt["gold_restaurant"]

    # Apply attack if configured
    items = _apply_attack_if_needed(items, attack_config, gold_id, req_id)

    # Apply shuffle based on gold position
    shuffled_items, mapping, shuffled_gold_pos = apply_shuffle(items, gold_idx, shuffle)

    # Format items as context
    context, item_count, coverage_stats = _format_context(
        shuffled_items, mode, token_budget, model
    )

    # Track per-request token usage
    tracker = get_usage_tracker()
    start_idx = len(tracker.get_records())

    # Invoke method and parse response
    shuffled_preds = []
    try:
        response = _invoke_method(method, query, context, k, req_id)
        shuffled_preds = parse_indices(response, item_count, k)
    except Exception as e:
        error_str = str(e)
        if "context_length_exceeded" in error_str or "too many tokens" in error_str.lower():
            raise ContextLengthExceeded(error_str)
        print(f"[ERROR] {req_id}: {type(e).__name__}: {error_str[:200]}", flush=True)

    # Map predictions back to original indices
    pred_indices = unmap_predictions(shuffled_preds, mapping)

    # Build result with usage stats
    usage_records = tracker.get_records()[start_idx:]
    return _build_result(
        req_id, pred_indices, shuffled_preds, gold_idx, shuffled_gold_pos,
        gt, usage_records, coverage_stats
    )


def _run_with_progress(generator, has_rich_display: bool, description: str, total: int):
    """Run a generator with optional progress display."""
    if has_rich_display:
        # anot handles its own Rich live display
        for _ in generator:
            pass
    else:
        # Show rich progress bar for non-anot methods
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
                     shuffle: str = "random", parallel: bool = True,
                     max_workers: int = 40, attack_config: dict = None,
                     token_budget: int = None, model: str = None) -> dict:
    """Evaluate using ranking (Hits@K accuracy).

    Args:
        items: All items (restaurants) to rank
        method: LLM method that returns ranking via evaluate_ranking()
        requests: List of requests
        groundtruth: {request_id: {"gold_restaurant": str, "gold_idx": int}}
        mode: "string" or "dict" for formatting
        k: Number of top predictions to check (default 5 for Hits@5)
        shuffle: Shuffle strategy ("none", "middle", "random") - default "random"
        parallel: Whether to use parallel execution (default True)
        max_workers: Maximum number of worker threads (default 40)
        attack_config: Optional attack configuration for per-request attacks
        token_budget: Optional token budget for pack-to-budget truncation (string mode only)
        model: Model name for tokenizer (required if token_budget is set)

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
            items, method, requests, groundtruth, mode, k, shuffle, parallel, max_workers,
            has_rich_display, attack_config, token_budget, model
        )
    finally:
        if has_rich_display:
            method.stop_display()


def _evaluate_ranking_inner(items, method, requests, groundtruth, mode, k, shuffle, parallel, max_workers, has_rich_display, attack_config=None, token_budget=None, model=None):
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
                        req.get("context") or req.get("text", ""),  # query = user request
                        k, req, groundtruth, attack_config, token_budget, model
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
                query = req.get("context") or req.get("text", "")  # query = user request
                try:
                    result = evaluate_ranking_single(
                        method, items, mode, shuffle, query, k, req, groundtruth,
                        attack_config, token_budget, model
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
