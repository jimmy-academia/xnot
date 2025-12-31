"""Methods package - evaluation methods for restaurant recommendation."""

from typing import Callable

from .base import BaseMethod
from .cot import ChainOfThought
from .react import ReAct
from .rnot import method as rnot
from .knot import create_method, method as knot, set_output_dir, set_defense_mode, set_current_ids
from .knot_v4 import KnowledgeNetworkOfThoughtV4
from .knot_v5 import KnowledgeNetworkOfThoughtV5, create_method as create_method_v5


def get_method(args, run_dir: str = None) -> Callable:
    """Get configured method callable from args.

    Args:
        args: Parsed command-line arguments with:
            - args.method: "cot", "not", "knot", "dummy"
            - args.mode: "string" or "dict"
            - args.knot_approach: "base", "voting", "iterative", "divide", "v4", "v5"
            - args.defense: bool
        run_dir: Run directory for logging

    Returns:
        Callable that takes (query, context) and returns int (-1, 0, 1)
    """
    name = args.method
    mode = args.mode
    approach = getattr(args, 'knot_approach', 'base')
    defense = args.defense

    if name == "cot":
        from .cot import ChainOfThought
        cot_instance = ChainOfThought(defense=defense, run_dir=run_dir)
        # Return ranking callable if ranking mode is enabled (default: True)
        if getattr(args, 'ranking', True):
            k = getattr(args, 'k', 1)
            return lambda q, c: cot_instance.evaluate_ranking(q, c, k)
        return cot_instance

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

    elif name == "react":
        from .react import ReAct
        react_instance = ReAct(defense=defense, run_dir=run_dir)
        if getattr(args, 'ranking', True):
            k = getattr(args, 'k', 1)
            return lambda q, c: react_instance.evaluate_ranking(q, c, k)
        return react_instance

    elif name == "dummy":
        return lambda query, context: 0

    else:
        raise ValueError(f"Unknown method: {name}")


__all__ = [
    'BaseMethod',
    'ChainOfThought',
    'rnot',
    'knot',
    'get_method',
    'create_method',
    'set_output_dir',
    'set_defense_mode',
    'set_current_ids',
    'KnowledgeNetworkOfThoughtV4',
    'KnowledgeNetworkOfThoughtV5',
]
