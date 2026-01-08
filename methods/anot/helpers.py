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

    # Reserved input variables that are always available (not steps)
    input_vars = {"query", "input", "items", "context"}

    # Build dependency graph
    step_deps = {}
    for idx, instr in steps:
        deps = extract_dependencies(instr)
        # Filter out reserved input variables - they're always available
        step_deps[idx] = deps - input_vars

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


# Default threshold for truncating string values
# Set to 12 to preserve hours like "10:30-14:30" (11 chars)
DEFAULT_LEAF_TRUNCATE = 12


def _format_value(v, truncate: int = DEFAULT_LEAF_TRUNCATE) -> str:
    """Recursively format a value for display.

    Value types:
    - str: truncate at threshold
    - dict: expand (recurse into values)
    - list: if contains dict → expand; else → treat as str (truncate)

    Args:
        v: Value to format
        truncate: Max chars for str values (0 = no truncation)

    Returns:
        Formatted string representation
    """
    if v is None:
        return "None"

    # dict → expand
    if isinstance(v, dict):
        if not v:
            return "{}"
        parts = [f"{k}:{_format_value(dv, truncate)}" for k, dv in v.items()]
        return "{" + ",".join(parts) + "}"

    # list → check if contains dict
    if isinstance(v, list):
        if not v:
            return "[]"
        if any(isinstance(item, dict) for item in v):
            # contains dict → expand
            parts = [_format_value(item, truncate) for item in v]
            return "[" + ",".join(parts) + "]"
        else:
            # no dict → treat as str
            s = str(v)
            if truncate > 0 and len(s) > truncate:
                s = s[:truncate] + "..."
            return s

    # str (and other primitives) → truncate
    s = str(v).strip()
    if truncate > 0 and len(s) > truncate:
        s = s[:truncate] + "..."
    return s


def filter_fields(item: dict, drop_keys: set = None, drop_paths: set = None) -> dict:
    """Recursively filter fields from an item dict.

    Task-specific function to remove internal IDs or unnecessary fields.

    Args:
        item: Item dict to filter
        drop_keys: Keys to drop at any level (e.g., {'business_id', 'user_id'})
        drop_paths: Dot-notation paths to drop (e.g., {'reviews.user.friends'})

    Returns:
        Filtered copy of the item
    """
    drop_keys = drop_keys or set()
    drop_paths = drop_paths or set()

    def _filter(obj, path=""):
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                current_path = f"{path}.{k}" if path else k
                # Skip if key in drop_keys or path in drop_paths
                if k in drop_keys:
                    continue
                if current_path in drop_paths:
                    continue
                result[k] = _filter(v, current_path)
            return result
        elif isinstance(obj, list):
            return [_filter(item, path) for item in obj]
        else:
            return obj

    return _filter(item)


# Default fields to drop for restaurant ranking task
RESTAURANT_DROP_KEYS = {'business_id', 'review_id', 'user_id'}
# Drop raw friends IDs (will be replaced with names), keep elite
RESTAURANT_DROP_PATHS = {'reviews.user.friends'}


def _load_social_mapping(data_name: str):
    """Load synthesized social mapping for friend name enrichment."""
    import json
    from data.loader import DATA_DIR
    mapping_path = DATA_DIR / data_name / "user_mapping.json"
    if not mapping_path.exists():
        return None
    with open(mapping_path) as f:
        return json.load(f)


def _enrich_with_friend_names(items: list, data_name: str) -> list:
    """Add friend names to reviews based on synthesized social mapping.

    For each review, if the reviewer's name matches a user in the mapping,
    add their friends' names as 'friend_names' field.
    """
    mapping = _load_social_mapping(data_name)
    if not mapping:
        return items

    user_names = mapping.get("user_names", {})
    friend_graph = mapping.get("friend_graph", {})

    # Build reverse lookup: name -> user_id
    name_to_id = {name: uid for uid, name in user_names.items()}

    enriched = []
    for item in items:
        item = dict(item)  # shallow copy
        if "reviews" in item:
            new_reviews = []
            for review in item["reviews"]:
                review = dict(review)  # shallow copy
                if "user" in review:
                    user = dict(review["user"])
                    reviewer_name = user.get("name", "")

                    # Look up reviewer in synthesized mapping
                    reviewer_id = name_to_id.get(reviewer_name)
                    if reviewer_id and reviewer_id in friend_graph:
                        friend_ids = friend_graph[reviewer_id]
                        friend_names = [user_names.get(fid, "") for fid in friend_ids]
                        friend_names = [n for n in friend_names if n]  # filter empty
                        if friend_names:
                            user["friend_names"] = friend_names

                    review["user"] = user
                new_reviews.append(review)
            item["reviews"] = new_reviews
        enriched.append(item)

    return enriched


def filter_items_for_ranking(items: list, data_name: str = "philly_cafes") -> list:
    """Filter and enrich items for restaurant ranking task.

    1. Removes internal IDs (business_id, review_id, user_id)
    2. Removes raw friends IDs (replaced with friend_names)
    3. Enriches with friend names from synthesized social mapping
    """
    # First filter out IDs and raw friends
    filtered = [
        filter_fields(item, drop_keys=RESTAURANT_DROP_KEYS, drop_paths=RESTAURANT_DROP_PATHS)
        for item in items
    ]
    # Then enrich with friend names
    return _enrich_with_friend_names(filtered, data_name)


def format_items_compact(items: list, truncate: int = DEFAULT_LEAF_TRUNCATE) -> str:
    """Format a list of items with full schema structure for LLM analysis.

    General-purpose formatter that recursively expands any dict/list structure.
    Only leaf string values are truncated at the threshold.

    Args:
        items: List of item dicts (any structure)
        truncate: Max chars for leaf string values (default 12, 0 = no truncation)

    Returns:
        Formatted string with one item per block, 1-indexed

    Example output:
        Item 1: {name:Milkcrate...,attributes:{Ambience:{hipster:True,casual:True}},reviews:[{text:Great cof...},...]}
    """
    lines = []
    for i, item in enumerate(items):
        formatted = _format_value(item, truncate)
        lines.append(f"Item {i+1}: {formatted}")

    return "\n".join(lines)


def format_items_attrs_only(items: list, truncate: int = DEFAULT_LEAF_TRUNCATE) -> str:
    """Format items with ONLY attributes/hours (no reviews) for attribute checks.

    This drastically reduces token count for conditions that don't need review data.
    For N=50 items, this can reduce from ~400k tokens to ~20k tokens.

    Args:
        items: List of item dicts
        truncate: Max chars for leaf string values (default 12)

    Returns:
        Formatted string with one item per block, 1-indexed, reviews stripped
    """
    lines = []
    for i, item in enumerate(items):
        # Create copy without reviews
        item_no_reviews = {k: v for k, v in item.items() if k != 'reviews'}
        formatted = _format_value(item_no_reviews, truncate)
        lines.append(f"Item {i+1}: {formatted}")

    return "\n".join(lines)


def format_schema_compact(items: list, num_examples: int = 2, truncate: int = 50) -> str:
    """Format 1-2 example items with FULL structure to show available schema.

    Unlike format_items_compact, this uses longer truncation (50 chars) to show
    more detail, helping LLM understand what fields are available.

    Args:
        items: List of item dicts
        num_examples: How many example items to show (default 2)
        truncate: Max chars for leaf string values (default 50 - more detail)

    Returns:
        Formatted string with schema examples

    Example output:
        [EXAMPLE ITEM 1]
        {name:Milkcrate Cafe,attributes:{GoodForKids:True,WiFi:free,Ambience:{hipster:True,casual:True,...}},hours:{Monday:8:0-18:0,Tuesday:8:0-18:0,...},reviews:[{text:Great coffee and atmosphere...,user:{name:John,elite:[2019,2020]}}]}

        [EXAMPLE ITEM 2]
        ...
    """
    lines = []
    for i, item in enumerate(items[:num_examples]):
        formatted = _format_value(item, truncate)
        lines.append(f"[EXAMPLE ITEM {i+1}]")
        lines.append(formatted)
        lines.append("")  # blank line between items

    return "\n".join(lines)


# =============================================================================
# Multi-step Phase 1 Parsing Functions
# =============================================================================

def parse_conditions(response: str) -> list:
    """Parse Step 1 output: extract conditions from LLM response.

    Args:
        response: LLM response with lines like "[ATTR] has drive-thru"
                  or "[REVIEW:POSITIVE] coffee"

    Returns:
        List of dicts: [{"type": "ATTR", "description": "has drive-thru"}, ...]
        For review sentiment: [{"type": "REVIEW", "sentiment": "positive", "description": "coffee"}, ...]
    """
    conditions = []
    for line in response.strip().split('\n'):
        line = line.strip()
        # Match [TYPE:SUBTYPE] description or [TYPE] description patterns
        match = re.match(r'\[(\w+)(?::(\w+))?\]\s*(.+)', line)
        if match:
            cond_type = match.group(1).upper()
            cond_subtype = match.group(2).upper() if match.group(2) else None
            description = match.group(3).strip()

            cond = {"type": cond_type, "description": description}
            if cond_subtype:
                # Handle [REVIEW:POSITIVE] or [REVIEW:NEGATIVE]
                cond["sentiment"] = cond_subtype.lower()  # "positive" or "negative"
            conditions.append(cond)
    return conditions


def parse_resolved_path(response: str) -> dict:
    """Parse Step 2 output: extract path resolution from LLM response.

    Args:
        response: LLM response with PATH, EXPECTED, TYPE lines

    Returns:
        Dict: {"path": "attributes.X", "expected": True, "type": "HARD"}
        or {"path": "reviews", "expected": "keyword", "type": "SOFT"}
    """
    result = {"path": None, "expected": None, "type": "HARD"}

    for line in response.strip().split('\n'):
        line = line.strip()
        if line.upper().startswith('PATH:'):
            result["path"] = line.split(':', 1)[1].strip()
        elif line.upper().startswith('EXPECTED:'):
            val = line.split(':', 1)[1].strip()
            # Parse boolean/string values
            if val.lower() == 'true':
                result["expected"] = True
            elif val.lower() == 'false':
                result["expected"] = False
            else:
                # Remove quotes if present
                result["expected"] = val.strip('"\'')
        elif line.upper().startswith('TYPE:'):
            result["type"] = line.split(':', 1)[1].strip().upper()

    # Normalize SOFT type variations
    if result["type"] in ("SOFT", "REVIEW_SEARCH", "REVIEW"):
        result["type"] = "SOFT"
    else:
        result["type"] = "HARD"

    return result


def parse_candidates(response: str) -> list:
    """Parse Step 3 output: extract candidate item numbers.

    Args:
        response: LLM response with ===CANDIDATES=== section

    Returns:
        List of int: [2, 4, 6, 7, 8, 9, 10]
    """
    candidates = []

    # Look for ===CANDIDATES=== section
    if '===CANDIDATES===' in response:
        parts = response.split('===CANDIDATES===')
        if len(parts) > 1:
            candidates_section = parts[1].split('===')[0]  # Get until next section
            # Extract numbers
            numbers = re.findall(r'\d+', candidates_section)
            candidates = [int(n) for n in numbers]
    else:
        # Fallback: look for list of numbers anywhere
        # Try to find a bracketed list [1, 2, 3]
        match = re.search(r'\[[\d,\s]+\]', response)
        if match:
            numbers = re.findall(r'\d+', match.group())
            candidates = [int(n) for n in numbers]

    return candidates


def parse_lwt_skeleton(response: str) -> list:
    """Parse Step 4 output: extract LWT skeleton steps.

    Args:
        response: LLM response with ===LWT_SKELETON=== section

    Returns:
        List of tuples: [("r2", "LLM(...)"), ("r4", "LLM(...)"), ("final", "LLM(...)")]
    """
    steps = []

    # Look for ===LWT_SKELETON=== section
    content = response
    if '===LWT_SKELETON===' in response:
        content = response.split('===LWT_SKELETON===')[1]
        # Stop at next section marker if present
        if '===' in content:
            content = content.split('===')[0]

    # Match (var_name)=LLM('...') patterns
    # Pattern handles nested parens and quotes
    pattern = r"\((\w+)\)\s*=\s*LLM\s*\(\s*['\"](.+?)['\"]\s*\)"
    for match in re.finditer(pattern, content, re.DOTALL):
        var_name = match.group(1)
        prompt = match.group(2)
        steps.append((var_name, f"LLM('{prompt}')"))

    return steps


def format_items_for_ruleout(items: list, hard_conditions: list) -> str:
    """Format items with only relevant attributes for quick rule-out.

    Args:
        items: List of item dicts
        hard_conditions: List of resolved condition dicts with 'path' field

    Returns:
        Compact string showing only relevant attributes per item
    """
    # Extract field names from paths
    fields_needed = set()
    for cond in hard_conditions:
        path = cond.get("path", "")
        # Extract top-level field from path like "attributes.GoodForKids"
        if path.startswith("attributes."):
            # Get the specific attribute
            attr_path = path[len("attributes."):]
            # Handle nested like "Ambience.hipster"
            attr_name = attr_path.split('.')[0]
            fields_needed.add(attr_name)
        elif path == "hours" or path.startswith("hours."):
            fields_needed.add("hours")

    lines = []
    for i, item in enumerate(items):
        attrs = item.get("attributes", {})
        hours = item.get("hours", {})

        # Build compact representation
        parts = []
        for field in sorted(fields_needed):
            if field == "hours":
                val = hours if hours else None
            elif field in attrs:
                val = attrs[field]
            else:
                val = None

            # Format value compactly
            if val is None:
                parts.append(f"{field}=None")
            elif isinstance(val, dict):
                # For nested dicts like Ambience, show relevant subfields
                sub_parts = []
                for cond in hard_conditions:
                    path = cond.get("path", "")
                    if f"attributes.{field}." in path:
                        subfield = path.split('.')[-1]
                        sub_val = val.get(subfield)
                        sub_parts.append(f"{subfield}={sub_val}")
                if sub_parts:
                    parts.append(f"{field}={{{','.join(sub_parts)}}}")
                else:
                    parts.append(f"{field}={val}")
            elif isinstance(val, bool):
                parts.append(f"{field}={val}")
            else:
                parts.append(f"{field}={repr(val)}")

        lines.append(f"Item {i+1}: {', '.join(parts)}")

    return "\n".join(lines)


def get_attr_value(item: dict, path: str):
    """Get value from item using dot-notation path.

    Args:
        item: Item dict
        path: Path like "attributes.GoodForKids" or "attributes.Ambience.hipster"

    Returns:
        Value at path or None if not found
    """
    parts = path.split('.')
    current = item
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current
