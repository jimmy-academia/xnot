"""
G1a-ANoT: Adaptive Network of Thought for Peanut Allergy Safety Task

3-Phase LWT Architecture:
- Phase 1: Understand formula → Create LWT seed
- Phase 2: Apply seed to context → Form concrete script
- Phase 3: Execute script → Compute primitives
"""

from .core import G1aANoT
from .prompts import PHASE1_PROMPT, PHASE2_PROMPT, EXTRACTION_PROMPT
from .tools import keyword_search, get_review_metadata, get_cuisine_modifier
from .helpers import parse_lwt_script, build_execution_layers, substitute_variables

__all__ = [
    'G1aANoT',
    'PHASE1_PROMPT',
    'PHASE2_PROMPT',
    'EXTRACTION_PROMPT',
    'keyword_search',
    'get_review_metadata',
    'get_cuisine_modifier',
    'parse_lwt_script',
    'build_execution_layers',
    'substitute_variables',
]
