#!/usr/bin/env python3
"""Shuffle utilities for ranking evaluation."""


def shuffle_gold_to_middle(items: list, gold_idx: int) -> tuple[list, dict]:
    """Move gold item to middle position.

    Args:
        items: List of items
        gold_idx: Original index of gold item (0-indexed)

    Returns:
        (shuffled_items, mapping) where mapping[shuffled_pos] = original_idx
    """
    n = len(items)
    middle = n // 2

    # Build new order: remove gold, insert at middle
    indices = list(range(n))
    indices.remove(gold_idx)
    indices.insert(middle, gold_idx)

    shuffled = [items[i] for i in indices]
    mapping = {new_pos: orig_idx for new_pos, orig_idx in enumerate(indices)}
    return shuffled, mapping


def apply_shuffle(items: list, gold_idx: int, strategy: str) -> tuple[list, dict, int]:
    """Apply shuffle strategy.

    Args:
        items: List of items
        gold_idx: Original index of gold item (0-indexed)
        strategy: "none", "middle", or "random"

    Returns:
        (shuffled_items, mapping, shuffled_gold_pos) where:
        - mapping[shuffled_pos] = original_idx
        - shuffled_gold_pos = position of gold in shuffled list
    """
    if strategy == "none":
        mapping = {i: i for i in range(len(items))}
        return items, mapping, gold_idx
    elif strategy == "middle":
        shuffled, mapping = shuffle_gold_to_middle(items, gold_idx)
        shuffled_gold_pos = [k for k, v in mapping.items() if v == gold_idx][0]
        return shuffled, mapping, shuffled_gold_pos
    elif strategy == "random":
        import random
        indices = list(range(len(items)))
        random.shuffle(indices)
        shuffled = [items[i] for i in indices]
        mapping = {new_pos: orig_idx for new_pos, orig_idx in enumerate(indices)}
        shuffled_gold_pos = [k for k, v in mapping.items() if v == gold_idx][0]
        return shuffled, mapping, shuffled_gold_pos
    else:
        # Default to no shuffle
        mapping = {i: i for i in range(len(items))}
        return items, mapping, gold_idx


def unmap_predictions(pred_indices: list[int], mapping: dict) -> list[int]:
    """Convert shuffled predictions back to original indices.

    Args:
        pred_indices: Predictions in shuffled space (1-indexed)
        mapping: mapping[shuffled_pos] = original_idx (0-indexed)

    Returns:
        Predictions in original space (1-indexed)
    """
    result = []
    for p in pred_indices:
        shuffled_pos = p - 1  # Convert to 0-indexed
        if shuffled_pos in mapping:
            result.append(mapping[shuffled_pos] + 1)  # Back to 1-indexed
    return result
