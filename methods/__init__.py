"""Methods package - evaluation methods for restaurant recommendation."""

import os
from typing import Callable

from .base import BaseMethod
from .rnot import method as rnot


# Registry of standard methods: {name: (module, class_name, supports_defense)}
METHOD_REGISTRY = {
    "cot": ("cot", "ChainOfThought", True),
    "react": ("react", "ReAct", True),
    "cotsc": ("cotsc", "CoTSelfConsistency", False),
    "l2m": ("l2m", "LeastToMost", False),
    "ps": ("ps", "PlanAndSolve", False),
    "pal": ("pal", "ProgramAidedLanguage", True),
    "pot": ("pot", "ProgramOfThoughts", True),
    "cot_table": ("cot_table", "ChainOfTable", True),
    "selfask": ("selfask", "SelfAsk", False),
    "parade": ("parade", "PaRaDe", False),
    "decomp": ("decomp", "DecomposedPrompting", True),
    "finegrained": ("finegrained", "FineGrainedRanker", True),
    "rankgpt": ("rankgpt", "RankGPT", False),
    "prp": ("prp", "PairwiseRankingPrompting", True),
    "setwise": ("setwise", "Setwise", False),
    "listwise": ("listwise", "ListwiseRanker", True),
    "weaver": ("weaver", "Weaver", True),
}


def _create_standard_method(name: str, defense: bool, run_dir: str):
    """Create a standard method from the registry."""
    module_name, class_name, supports_defense = METHOD_REGISTRY[name]
    module = __import__(f"methods.{module_name}", fromlist=[class_name])
    cls = getattr(module, class_name)

    # Pass defense only if the method supports it
    if supports_defense:
        return cls(defense=defense, run_dir=run_dir)
    return cls(run_dir=run_dir)


def _wrap_for_ranking(instance, args):
    """Wrap method instance for ranking if enabled and supported."""
    if getattr(args, 'ranking', True) and hasattr(instance, 'evaluate_ranking'):
        k = getattr(args, 'k', 1)
        return lambda q, c: instance.evaluate_ranking(q, c, k)
    return instance


def get_method(args, run_dir: str = None) -> Callable:
    """Get configured method callable from args.

    Args:
        args: Parsed command-line arguments with:
            - args.method: "cot", "not", "anot", "react", "decomp", etc.
            - args.defense: bool
        run_dir: Run directory for logging

    Returns:
        Callable that takes (query, context) and returns int (-1, 0, 1)
    """
    name = args.method
    defense = args.defense

    # Special case: dummy method
    if name == "dummy":
        return lambda query, context: 0

    # Special case: rnot (module function, not class)
    if name == "not":
        return rnot

    # Special case: anot methods with factory functions
    if name in ("anot", "anot_v3"):
        from .anot_v3 import create_method as create_anot_v3
        debug = os.environ.get("KNOT_DEBUG", "0") == "1"
        instance = create_anot_v3(run_dir=run_dir, defense=defense, debug=debug)
        return _wrap_for_ranking(instance, args)

    if name == "anot_origin":
        from .anot_origin import create_method as create_anot_origin
        debug = os.environ.get("KNOT_DEBUG", "0") == "1"
        instance = create_anot_origin(run_dir=run_dir, defense=defense, debug=debug)
        return _wrap_for_ranking(instance, args)

    # Standard methods from registry
    if name in METHOD_REGISTRY:
        instance = _create_standard_method(name, defense, run_dir)
        return _wrap_for_ranking(instance, args)

    raise ValueError(f"Unknown method: {name}")


# Re-export commonly used classes for direct import
from .cot import ChainOfThought
from .react import ReAct
from .decomp import DecomposedPrompting
from .finegrained import FineGrainedRanker
from .prp import PairwiseRankingPrompting
from .listwise import ListwiseRanker
from .anot_v3 import AdaptiveNetworkOfThoughtV3, create_method as create_method_anot_v3
from .anot_origin import AdaptiveNetworkOfThoughtOrigin, create_method as create_method_anot_origin


__all__ = [
    'BaseMethod',
    'ChainOfThought',
    'ReAct',
    'DecomposedPrompting',
    'FineGrainedRanker',
    'PairwiseRankingPrompting',
    'ListwiseRanker',
    'rnot',
    'get_method',
    'AdaptiveNetworkOfThoughtV3',
    'create_method_anot_v3',
    'AdaptiveNetworkOfThoughtOrigin',
    'create_method_anot_origin',
    'METHOD_REGISTRY',
]
