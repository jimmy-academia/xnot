"""Methods package - evaluation methods for restaurant recommendation."""

from typing import Callable

from .base import BaseMethod
from .cot import ChainOfThought
from .react import ReAct
from .decomp import DecomposedPrompting
from .finegrained import FineGrainedRanker
from .prp import PairwiseRankingPrompting
from .listwise import ListwiseRanker
from .rnot import method as rnot
from .anot import AdaptiveNetworkOfThought, create_method as create_method_anot


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

    elif name == "anot":
        from .anot import create_method as create_anot
        import os
        debug = os.environ.get("KNOT_DEBUG", "0") == "1"
        return create_anot(run_dir=run_dir, debug=debug)

    elif name == "react":
        from .react import ReAct
        react_instance = ReAct(defense=defense, run_dir=run_dir)
        if getattr(args, 'ranking', True):
            k = getattr(args, 'k', 1)
            return lambda q, c: react_instance.evaluate_ranking(q, c, k)
        return react_instance

    elif name == "cotsc":
        from .cotsc import CoTSelfConsistency
        cotsc_instance = CoTSelfConsistency(run_dir=run_dir)
        if getattr(args, 'ranking', True):
            k = getattr(args, 'k', 1)
            return lambda q, c: cotsc_instance.evaluate_ranking(q, c, k)
        return cotsc_instance

    elif name == "l2m":
        from .l2m import LeastToMost
        l2m_instance = LeastToMost(run_dir=run_dir)
        if getattr(args, 'ranking', True):
            k = getattr(args, 'k', 1)
            return lambda q, c: l2m_instance.evaluate_ranking(q, c, k)
        return l2m_instance

    elif name == "ps":
        from .ps import PlanAndSolve
        ps_instance = PlanAndSolve(run_dir=run_dir)
        if getattr(args, 'ranking', True):
            k = getattr(args, 'k', 1)
            return lambda q, c: ps_instance.evaluate_ranking(q, c, k)
        return ps_instance

    elif name == "selfask":
        from .selfask import SelfAsk
        selfask_instance = SelfAsk(run_dir=run_dir)
        if getattr(args, 'ranking', True):
            k = getattr(args, 'k', 1)
            return lambda q, c: selfask_instance.evaluate_ranking(q, c, k)
        return selfask_instance

    elif name == "parade":
        from .parade import PaRaDe
        parade_instance = PaRaDe(run_dir=run_dir)
        if getattr(args, 'ranking', True):
            k = getattr(args, 'k', 1)
            return lambda q, c: parade_instance.evaluate_ranking(q, c, k)
        return parade_instance

    elif name == "decomp":
        from .decomp import DecomposedPrompting
        decomp_instance = DecomposedPrompting(defense=defense, run_dir=run_dir)
        if getattr(args, 'ranking', True):
            k = getattr(args, 'k', 1)
            return lambda q, c: decomp_instance.evaluate_ranking(q, c, k)
        return decomp_instance

    elif name == "finegrained":
        from .finegrained import FineGrainedRanker
        fg_instance = FineGrainedRanker(defense=defense, run_dir=run_dir)
        if getattr(args, 'ranking', True):
            k = getattr(args, 'k', 1)
            return lambda q, c: fg_instance.evaluate_ranking(q, c, k)
        return fg_instance

    elif name == "rankgpt":
        from .rankgpt import RankGPT
        rankgpt_instance = RankGPT(run_dir=run_dir)
        if getattr(args, 'ranking', True):
            k = getattr(args, 'k', 1)
            return lambda q, c: rankgpt_instance.evaluate_ranking(q, c, k)
        return rankgpt_instance

    elif name == "prp":
        from .prp import PairwiseRankingPrompting
        prp_instance = PairwiseRankingPrompting(defense=defense, run_dir=run_dir)
        if getattr(args, 'ranking', True):
            k = getattr(args, 'k', 1)
            return lambda q, c: prp_instance.evaluate_ranking(q, c, k)
        return prp_instance

    elif name == "setwise":
        from .setwise import Setwise
        setwise_instance = Setwise(run_dir=run_dir)
        if getattr(args, 'ranking', True):
            k = getattr(args, 'k', 1)
            return lambda q, c: setwise_instance.evaluate_ranking(q, c, k)
        return setwise_instance

    elif name == "listwise":
        from .listwise import ListwiseRanker
        listwise_instance = ListwiseRanker(defense=defense, run_dir=run_dir)
        if getattr(args, 'ranking', True):
            k = getattr(args, 'k', 1)
            return lambda q, c: listwise_instance.evaluate_ranking(q, c, k)
        return listwise_instance

    elif name == "dummy":
        return lambda query, context: 0

    else:
        raise ValueError(f"Unknown method: {name}")


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
    'AdaptiveNetworkOfThought',
    'create_method_anot',
]
