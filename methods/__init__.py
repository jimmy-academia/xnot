"""Methods package - evaluation methods for restaurant recommendation."""

from typing import Callable

from .cot import method as cot
from .rnot import method as rnot
from .knot import create_method, method as knot, set_output_dir, set_defense_mode, set_current_ids
from .knot_v4 import KnowledgeNetworkOfThoughtV4
from .knot_v5 import KnowledgeNetworkOfThoughtV5, create_method as create_method_v5


def get_method(name: str, mode: str = "string", approach: str = "base",
               defense: bool = False, run_dir: str = None) -> Callable:
    """Get configured method callable.

    Args:
        name: Method name ("cot", "not", "knot", "dummy")
        mode: Input mode for knot ("string" or "dict")
        approach: KNoT approach ("base", "voting", "iterative", "divide", "v4")
        defense: Enable defense mode
        run_dir: Run directory for logging (knot only)

    Returns:
        Callable that takes (query, context) and returns int (-1, 0, 1)
    """
    if name == "cot":
        from .cot import method, set_defense_mode as cot_set_defense
        if defense:
            cot_set_defense(True)
        return method

    elif name == "not":
        from .rnot import method
        return method

    elif name == "knot":
        # v5 is a separate implementation with built-in defense
        if approach == "v5":
            from .knot_v5 import create_method as create_v5
            import os
            debug = os.environ.get("KNOT_DEBUG", "0") == "1"
            return create_v5(run_dir=run_dir, debug=debug)

        from .knot import create_method, set_output_dir, set_defense_mode as knot_set_defense
        if run_dir:
            set_output_dir(run_dir)
        knot_set_defense(defense)
        return create_method(mode=mode, approach=approach, run_dir=run_dir)

    elif name == "dummy":
        return lambda query, context: 0

    else:
        raise ValueError(f"Unknown method: {name}")


__all__ = [
    'cot',
    'rnot',
    'knot',
    'get_method',
    'create_method',
    'set_output_dir',
    'set_defense_mode',
    'set_current_ids',
    'KnowledgeNetworkOfThoughtV4',
]
