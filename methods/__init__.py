"""Method registry for the evaluation framework."""

from typing import Callable

from .base import BaseMethod
from .cot import ChainOfThought
from .ps import PlanAndSolve
from .plan_act import PlanAndAct
from .listwise import ListwiseRanker
from .weaver import Weaver
from .anot import AdaptiveNetworkOfThought, create_method as create_anot

# Additional baseline methods
from .l2m import LeastToMost
from .selfask import SelfAsk
from .rankgpt import RankGPT
from .setwise import Setwise
from .parade import PaRaDe
from .react import ReAct
from .decomp import DecomposedPrompting
from .pal import ProgramAidedLanguage
from .pot import ProgramOfThoughts
from .cot_table import ChainOfTable
from .finegrained import FineGrainedRanker
from .prp import PairwiseRankingPrompting


# Method registry: name -> (class, supports_defense)
METHOD_REGISTRY = {
    # Core methods
    "cot": (ChainOfThought, True),
    "ps": (PlanAndSolve, False),
    "plan_act": (PlanAndAct, True),
    "listwise": (ListwiseRanker, True),
    "weaver": (Weaver, True),
    "anot": (AdaptiveNetworkOfThought, True),
    # CoT variants
    "l2m": (LeastToMost, False),
    "selfask": (SelfAsk, False),
    # Program-aided methods
    "pal": (ProgramAidedLanguage, False),
    "pot": (ProgramOfThoughts, False),
    "cot_table": (ChainOfTable, False),
    # Ranking methods
    "rankgpt": (RankGPT, False),
    "setwise": (Setwise, False),
    "parade": (PaRaDe, False),
    "finegrained": (FineGrainedRanker, False),
    "prp": (PairwiseRankingPrompting, False),
    # Agentic methods
    "react": (ReAct, False),
    "decomp": (DecomposedPrompting, False),
}


class DummyMethod(BaseMethod):
    """Dummy method for testing - always returns 0."""

    name = "dummy"

    def evaluate(self, query, context: str) -> int:
        return 0

    def evaluate_ranking(self, query: str, context: str, k: int = 1) -> str:
        return "1"


def get_method(name: str, run_dir: str = None, defense: bool = False, verbose: bool = True, **kwargs) -> BaseMethod:
    """Get a method instance by name.

    Args:
        name: Method name (cot, ps, listwise, weaver, anot, dummy)
        run_dir: Optional run directory for logging
        defense: Whether to enable defense mode
        verbose: Whether to enable verbose output (default: True)
        **kwargs: Additional arguments passed to method constructor

    Returns:
        BaseMethod instance

    Raises:
        ValueError: If method name is not recognized
    """
    if name == "dummy":
        return DummyMethod(run_dir=run_dir, defense=defense, verbose=verbose, **kwargs)

    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {name}. Available: {list(METHOD_REGISTRY.keys()) + ['dummy']}")

    method_class, supports_defense = METHOD_REGISTRY[name]

    # Only pass defense if method supports it
    if supports_defense:
        return method_class(run_dir=run_dir, defense=defense, verbose=verbose, **kwargs)
    else:
        return method_class(run_dir=run_dir, verbose=verbose, **kwargs)


def list_methods() -> list:
    """List all available method names."""
    return list(METHOD_REGISTRY.keys()) + ["dummy"]


__all__ = [
    # Base
    "BaseMethod",
    "DummyMethod",
    # Core methods
    "ChainOfThought",
    "PlanAndSolve",
    "PlanAndAct",
    "ListwiseRanker",
    "Weaver",
    "AdaptiveNetworkOfThought",
    # CoT variants
    "LeastToMost",
    "SelfAsk",
    # Program-aided methods
    "ProgramAidedLanguage",
    "ProgramOfThoughts",
    "ChainOfTable",
    # Ranking methods
    "RankGPT",
    "Setwise",
    "PaRaDe",
    "FineGrainedRanker",
    "PairwiseRankingPrompting",
    # Agentic methods
    "ReAct",
    "DecomposedPrompting",
    # Utilities
    "get_method",
    "list_methods",
    "METHOD_REGISTRY",
]
