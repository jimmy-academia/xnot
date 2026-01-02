"""Method registry for the evaluation framework."""

from typing import Callable

from .base import BaseMethod
from .cot import ChainOfThought
from .ps import PlanAndSolve
from .plan_act import PlanAndAct
from .listwise import ListwiseRanker
from .weaver import Weaver
from .anot import AdaptiveNetworkOfThought, create_method as create_anot


# Method registry: name -> (class, supports_defense)
METHOD_REGISTRY = {
    "cot": (ChainOfThought, True),
    "ps": (PlanAndSolve, False),
    "plan_act": (PlanAndAct, True),
    "listwise": (ListwiseRanker, True),
    "weaver": (Weaver, True),
    "anot": (AdaptiveNetworkOfThought, True),
}


class DummyMethod(BaseMethod):
    """Dummy method for testing - always returns 0."""

    name = "dummy"

    def evaluate(self, query, context: str) -> int:
        return 0

    def evaluate_ranking(self, query: str, context: str, k: int = 1) -> str:
        return "1"


def get_method(name: str, run_dir: str = None, defense: bool = False, **kwargs) -> BaseMethod:
    """Get a method instance by name.

    Args:
        name: Method name (cot, ps, listwise, weaver, anot, dummy)
        run_dir: Optional run directory for logging
        defense: Whether to enable defense mode
        **kwargs: Additional arguments passed to method constructor

    Returns:
        BaseMethod instance

    Raises:
        ValueError: If method name is not recognized
    """
    if name == "dummy":
        return DummyMethod(run_dir=run_dir, defense=defense, **kwargs)

    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {name}. Available: {list(METHOD_REGISTRY.keys()) + ['dummy']}")

    method_class, supports_defense = METHOD_REGISTRY[name]

    # Only pass defense if method supports it
    if supports_defense:
        return method_class(run_dir=run_dir, defense=defense, **kwargs)
    else:
        return method_class(run_dir=run_dir, **kwargs)


def list_methods() -> list:
    """List all available method names."""
    return list(METHOD_REGISTRY.keys()) + ["dummy"]


__all__ = [
    "BaseMethod",
    "ChainOfThought",
    "PlanAndSolve",
    "PlanAndAct",
    "ListwiseRanker",
    "Weaver",
    "AdaptiveNetworkOfThought",
    "DummyMethod",
    "get_method",
    "list_methods",
    "METHOD_REGISTRY",
]
