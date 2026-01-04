#!/usr/bin/env python3
"""Helper functions for ANoT - dependency extraction, execution layers, formatting."""

import re


def extract_dependencies(instruction: str) -> set:
    """Extract step IDs referenced in instruction (e.g., {(0)}, {(5.agg)}, {(final)})."""
    matches = re.findall(r'\{\(([a-zA-Z0-9_.]+)\)\}', instruction)
    return set(matches)


def build_execution_layers(steps: list) -> list:
    """Group steps into layers that can run in parallel.

    Returns list of layers, where each layer is [(idx, instr), ...].
    Steps in the same layer have no dependencies on each other.

    Raises:
        ValueError: If a cycle is detected in LWT dependencies.
    """
    if not steps:
        return []

    # Build dependency graph
    step_deps = {}
    for idx, instr in steps:
        step_deps[idx] = extract_dependencies(instr)

    # Assign steps to layers using topological sort
    layers = []
    assigned = set()

    while len(assigned) < len(steps):
        current_layer = []
        for idx, instr in steps:
            if idx in assigned:
                continue
            deps = step_deps[idx]
            if deps <= assigned:
                current_layer.append((idx, instr))

        if not current_layer:
            remaining = [(idx, instr) for idx, instr in steps if idx not in assigned]
            if remaining:
                unresolved = [idx for idx, _ in remaining]
                raise ValueError(f"Cycle detected in LWT dependencies. Unresolved steps: {unresolved}")
            break

        layers.append(current_layer)
        for idx, _ in current_layer:
            assigned.add(idx)

    return layers


def format_items_compact(items: list) -> str:
    """Format items as one line each with key=value pairs.

    Example output:
    Item 0: "Tria Cafe" - HasTV=False, GoodForKids=False, DriveThru=None, WiFi=free
    Item 1: "Front Street" - HasTV=False, GoodForKids=True, DriveThru=None, WiFi=free

    Rules:
    - One line per item
    - Include item name
    - Flatten attributes to key=value pairs
    - Use None for missing attributes
    - For complex nested values (dicts), just show key=<dict>
    """
    lines = []
    for i, item in enumerate(items):
        name = item.get("item_name", f"Item {i}")
        attrs = item.get("attributes", {})
        hours = item.get("hours", {})

        # Flatten attributes
        attr_parts = []
        for k, v in sorted(attrs.items()):
            # Simplify complex values
            if isinstance(v, dict):
                attr_parts.append(f"{k}=<dict>")
            elif isinstance(v, str) and len(v) > 20:
                attr_parts.append(f"{k}={v[:15]}...")
            else:
                attr_parts.append(f"{k}={v}")

        # Add hours summary if present
        if hours:
            days = list(hours.keys())
            if days:
                attr_parts.append(f"hours={','.join(days[:3])}...")

        attrs_str = ", ".join(attr_parts) if attr_parts else "(no attributes)"
        lines.append(f'Item {i}: "{name}" - {attrs_str}')

    return "\n".join(lines)
