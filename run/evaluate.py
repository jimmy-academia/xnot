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
    max_reviews: int | None = None,
) -> tuple:
    """Format items as context for the model.

    Args:
        items: List of items (already shuffled)
        mode: "string" or "dict" for formatting
        max_reviews: Max reviews per restaurant (string mode only)

    Returns:
        (context, item_count, coverage_stats) - coverage_stats is None for dict mode
    """
    context, item_count, coverage_stats = format_ranking_query(items, mode, max_reviews)
    return context, item_count, coverage_stats


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


def _is_context_length_error(error: Exception) -> bool:
    """Check if exception is a context length error."""
    error_str = str(error).lower()
    context_errors = [
        "context_length_exceeded", "too many tokens", "maximum context length",
        "context window", "token limit", "max_tokens", "input too long"
    ]
    return any(err in error_str for err in context_errors)


def evaluate_ranking_single(method, items: list, mode: str, shuffle: str,
                            query: str, k: int, req: dict,
                            groundtruth: dict, attack_config: dict = None,
                            preset_max_reviews: int = None, quiet: bool = False) -> dict | None:
    """Evaluate a single request (thread-safe helper).

    For string mode: implements adaptive truncation - on context exceeded,
    reduces reviews by 1 per restaurant and retries until success or no reviews left.

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
        preset_max_reviews: Pre-determined max reviews limit (skip truncation discovery)
        quiet: If True, suppress error prints (for rich display compatibility)

    Returns:
        Result dict or None if no ground truth

    Raises:
        ContextLengthExceeded: If context limit exceeded even with no reviews
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

    # Calculate max reviews across all items (for adaptive truncation)
    max_reviews_in_data = max(len(item.get("reviews", [])) for item in shuffled_items)

    # Track per-request token usage
    tracker = get_usage_tracker()
    start_idx = len(tracker.get_records())

    # For string mode: adaptive truncation with retry
    # For dict mode: single attempt (no truncation needed)
    # If preset_max_reviews is provided, use it directly (skip discovery)
    current_max_reviews = preset_max_reviews  # None = unlimited (all reviews)
    coverage_stats = None
    shuffled_preds = []
    truncation_retries = 0
    raw_response = ""  # Store raw model output for debugging
    error_msg = None  # Store error for quiet mode

    while True:
        # Format items as context (with current review limit for string mode)
        context, item_count, coverage_stats = _format_context(
            shuffled_items, mode, current_max_reviews
        )

        try:
            response = _invoke_method(method, query, context, k, req_id)
            shuffled_preds = parse_indices(response, item_count, k)
            raw_response = response  # Store for debugging
            break  # Success!

        except Exception as e:
            if not _is_context_length_error(e):
                # Not a context error - log and continue with empty predictions
                error_msg = f"{type(e).__name__}: {str(e)[:200]}"
                if not quiet:
                    print(f"[ERROR] {req_id}: {error_msg}", flush=True)
                break

            # Context exceeded - try reducing reviews (string mode only)
            if mode != "string":
                # Dict mode doesn't support truncation - propagate error
                error_msg = f"Context exceeded: {str(e)[:200]}"
                if not quiet:
                    print(f"[CONTEXT EXCEEDED] {req_id}: {str(e)[:300]}", flush=True)
                raise ContextLengthExceeded(str(e))

            # Calculate next review limit
            if current_max_reviews is None:
                # First failure: start with max_reviews - 1
                current_max_reviews = max_reviews_in_data - 1
            else:
                current_max_reviews -= 1

            truncation_retries += 1

            if current_max_reviews < 0:
                # No reviews left and still failing - give up
                error_msg = "Cannot fit even with 0 reviews"
                if not quiet:
                    print(f"[CONTEXT EXCEEDED] {req_id}: {error_msg}", flush=True)
                raise ContextLengthExceeded(str(e))

            if not quiet:
                print(f"[TRUNCATE] {req_id}: Retrying with max_reviews={current_max_reviews} (retry {truncation_retries})", flush=True)

    # Add truncation info to coverage stats
    if coverage_stats and truncation_retries > 0:
        coverage_stats["truncation_retries"] = truncation_retries
    # Store final max_reviews for subsequent requests (optimization)
    if coverage_stats:
        coverage_stats["final_max_reviews"] = current_max_reviews

    # Map predictions back to original indices
    pred_indices = unmap_predictions(shuffled_preds, mapping)

    # Build result with usage stats
    usage_records = tracker.get_records()[start_idx:]
    result = _build_result(
        req_id, pred_indices, shuffled_preds, gold_idx, shuffled_gold_pos,
        gt, usage_records, coverage_stats
    )
    result["raw_response"] = raw_response  # Include raw model output for debugging
    if error_msg:
        result["error"] = error_msg
    return result


def _run_with_progress(generator, has_rich_display: bool, description: str, total: int):
    """Run a generator with optional progress display."""
    if has_rich_display:
        # anot handles its own Rich live display
        try:
            for _ in generator:
                pass
        except KeyboardInterrupt:
            raise  # Re-raise to be caught by outer handler
    else:
        # Show rich progress bar for non-anot methods
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.fields[status]}"),
        ) as progress:
            task = progress.add_task(description, total=total, status="")
            for item in generator:
                # Extract response preview if available
                status = ""
                if isinstance(item, dict) and "raw_response" in item:
                    resp = item["raw_response"]
                    req_id = item.get("request_id", "?")
                    # Truncate to ~60 chars for display
                    preview = resp.replace("\n", " ")[:60]
                    if len(resp) > 60:
                        preview += "..."
                    status = f"[dim]{req_id}: {preview}[/dim]"
                progress.update(task, advance=1, status=status)


def evaluate_ranking(items: list[dict], method: Callable, requests: list[dict],
                     groundtruth: dict, mode: str = "string", k: int = 5,
                     shuffle: str = "random", parallel: bool = True,
                     max_workers: int = 40, attack_config: dict = None) -> dict:
    """Evaluate using ranking (Hits@K accuracy).

    For string mode: implements adaptive truncation - on context exceeded,
    reduces reviews by 1 per restaurant and retries until success.

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

    Returns:
        Dict with results and accuracy stats
    """
    # Start rich Live display if method supports it (e.g., ANoT)
    has_rich_display = hasattr(method, 'start_display') and hasattr(method, 'stop_display')
    if has_rich_display:
        title = f"ANoT: {len(items)} candidates, k={k}"
        method.start_display(title=title, total=len(requests), requests=requests)

    result = None
    try:
        result = _evaluate_ranking_inner(
            items, method, requests, groundtruth, mode, k, shuffle, parallel, max_workers,
            has_rich_display, attack_config
        )
        return result
    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user (Ctrl+C)")
        # Return partial results
        return {
            "results": [],
            "req_ids": [r["id"] for r in requests],
            "stats": {},
            "context_exceeded": False,
            "interrupted": True,
            "errors": [],
        }
    finally:
        if has_rich_display:
            method.stop_display()
            # Print errors after display stops (for clean output)
            if result and result.get("errors"):
                print(f"\n\u26a0\ufe0f  {len(result['errors'])} error(s) during evaluation:")
                for req_id, error in result["errors"]:
                    print(f"  [{req_id}] {error}")


def _evaluate_ranking_inner(items, method, requests, groundtruth, mode, k, shuffle, parallel, max_workers, has_rich_display, attack_config=None):
    """Inner implementation of evaluate_ranking (wrapped by display context)."""
    req_ids = [r["id"] for r in requests]
    context_exceeded = False
    results = []

    # For string mode + parallel: run 1 request first to discover stable max_reviews
    # This avoids wasteful parallel retries where all requests fail and retry together
    preset_max_reviews = None
    remaining_requests = requests

    if parallel and mode == "string" and len(requests) > 1:
        first_req = requests[0]
        query = first_req.get("context") or first_req.get("text", "")
        if not has_rich_display:
            print(f"[TRUNCATE] Running {first_req['id']} first to determine stable max_reviews...", flush=True)
        try:
            first_result = evaluate_ranking_single(
                method, items, mode, shuffle, query, k, first_req, groundtruth, attack_config,
                quiet=has_rich_display
            )
            if first_result:
                results.append(first_result)
                # Extract discovered max_reviews for remaining requests
                if "coverage" in first_result and first_result["coverage"]:
                    preset_max_reviews = first_result["coverage"].get("final_max_reviews")
                    if preset_max_reviews is not None and not has_rich_display:
                        print(f"[TRUNCATE] Discovered stable max_reviews={preset_max_reviews}, using for remaining {len(requests)-1} requests", flush=True)
        except ContextLengthExceeded:
            context_exceeded = True
            return {"results": results, "req_ids": req_ids, "stats": {}, "context_exceeded": True}

        remaining_requests = requests[1:]

    if parallel:
        _interrupted = False

        def run_eval():
            nonlocal context_exceeded, _interrupted
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                def make_task(r):
                    return lambda: evaluate_ranking_single(
                        method, items, mode, shuffle,
                        r.get("context") or r.get("text", ""),
                        k, r, groundtruth, attack_config, preset_max_reviews,
                        quiet=has_rich_display
                    )
                futures = {
                    executor.submit(make_task(req)): req
                    for req in remaining_requests
                }

                try:
                    for future in as_completed(futures):
                        if _interrupted:
                            break
                        try:
                            result = future.result()
                            if result:
                                results.append(result)
                        except ContextLengthExceeded:
                            context_exceeded = True
                            for f in futures:
                                f.cancel()
                            break
                        except Exception:
                            pass  # Log but continue
                        yield 1
                except KeyboardInterrupt:
                    _interrupted = True
                    # Cancel all pending futures
                    for f in futures:
                        f.cancel()
                    raise

        description = f"Ranking evaluation (parallel, {max_workers} workers)..."
    else:
        def run_eval():
            nonlocal context_exceeded
            for req in remaining_requests:
                query = req.get("context") or req.get("text", "")  # query = user request
                try:
                    result = evaluate_ranking_single(
                        method, items, mode, shuffle, query, k, req, groundtruth,
                        attack_config, preset_max_reviews, quiet=has_rich_display
                    )
                    if result:
                        results.append(result)
                        yield result  # Yield result for progress display
                    else:
                        yield None
                except ContextLengthExceeded:
                    context_exceeded = True
                    return

        description = "Ranking evaluation (sequential)..."

    _run_with_progress(run_eval(), has_rich_display, description, len(remaining_requests))

    stats = compute_multi_k_stats(results, k)

    # Collect errors from results for display
    errors = [(r["request_id"], r["error"]) for r in results if r.get("error")]

    return {
        "results": results,
        "req_ids": req_ids,
        "stats": stats,
        "context_exceeded": context_exceeded,
        "errors": errors,
    }
