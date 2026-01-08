#!/usr/bin/env python3
"""ANoT Original - Backup of Adaptive Network of Thought package.

Three-phase architecture:
1. PLANNING: Schema extraction + LLM calls (conditions -> pruning -> skeleton)
2. EXPANSION: ReAct-like LWT expansion with tools
3. EXECUTION: Pure LWT execution with async DAG
"""

from .core import AdaptiveNetworkOfThoughtOriginal, create_method

__all__ = ["AdaptiveNetworkOfThoughtOriginal", "create_method"]
