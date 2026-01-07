# Methods Architecture Guide

This document describes the methods architecture and conventions.

---

## Directory Structure

```
methods/
├── __init__.py      # Registry, get_method(), list_methods()
├── base.py          # BaseMethod abstract class
├── shared.py        # Defense preamble
├── anot/            # Package: multi-file method
│   ├── __init__.py
│   ├── core.py
│   ├── helpers.py
│   ├── prompts.py
│   └── tools.py
├── cot.py           # Single-file method
├── ps.py
├── listwise.py
├── weaver.py
├── react.py
└── ...              # Other methods
```

---

## Base Class

All methods inherit from `BaseMethod`:

```python
class BaseMethod(ABC):
    name: str = "base"

    def __init__(self, run_dir: str = None, defense: bool = False, verbose: bool = True, **kwargs):
        ...

    @abstractmethod
    def evaluate(self, query: Any, context: str) -> int:
        """Returns: 1 (recommend), 0 (neutral), -1 (not recommend)"""
        pass

    def evaluate_ranking(self, query: str, context: str, k: int = 1) -> str:
        """Returns: String with top-k indices (e.g., '3' or '3, 1, 5')"""
        return "1"  # Default implementation
```

**Reference**: [methods/base.py](../../methods/base.py)

---

## Method Registry

Methods are registered in `METHOD_REGISTRY`:

```python
METHOD_REGISTRY = {
    # Core methods
    "cot": (ChainOfThought, True),        # supports_defense=True
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
```

Second element indicates whether method supports `--defense` flag.

**Reference**: [methods/__init__.py](../../methods/__init__.py)

---

## Package vs Single File

### Single File (e.g., `cot.py`)

Use for simple, single-pass methods:
- Single LLM call
- No shared state between calls
- Inline prompts
- < 200 lines total

### Package (e.g., `anot/`)

Use for complex, multi-phase methods:
- Multiple phases/stages
- Shared state between phases
- Multiple helper functions
- Separate prompt templates
- > 200 lines total

**Package structure**:
```
anot/
├── __init__.py      # Re-exports main class
├── core.py          # Main class with phases
├── helpers.py       # Utility functions
├── prompts.py       # LLM prompt templates
└── tools.py         # Tool definitions (if applicable)
```

---

## String Mode vs Dict Mode

### String Mode

Methods receive context as formatted string. Pack-to-budget truncation applied.

**Methods**: cot, ps, plan_act, listwise, l2m, selfask, rankgpt, setwise, parade, finegrained, prp, decomp, pal, pot, cot_table

### Dict Mode

Methods receive context as full data dict. No truncation, selective access.

**Methods**: anot, weaver, react

Defined in `data/loader.py`:
```python
DICT_MODE_METHODS = {"anot", "weaver", "react"}
```

---

## Defense Support

Methods supporting `--defense` flag:
- cot
- plan_act
- listwise
- weaver
- anot

Defense preamble is added to system prompts. See [doc/reference/defense_mode.md](../reference/defense_mode.md).

---

## Method Categories

### Core Methods

| Method | Description |
|--------|-------------|
| cot | Chain-of-Thought reasoning |
| ps | Plan-and-Solve |
| plan_act | Plan then Act |
| listwise | Listwise reranking |
| weaver | SQL+LLM hybrid |
| anot | Adaptive Network of Thought (3-phase) |

### CoT Variants

| Method | Description |
|--------|-------------|
| l2m | Least-to-Most decomposition |
| selfask | Self-Ask with follow-up questions |

### Program-Aided Methods

| Method | Description |
|--------|-------------|
| pal | Program-Aided Language models |
| pot | Program of Thoughts |
| cot_table | Chain-of-Table |

### Ranking Methods

| Method | Description |
|--------|-------------|
| rankgpt | RankGPT permutation |
| setwise | Setwise comparison |
| parade | Pairwise Ranking Decomposition |
| finegrained | Fine-grained relevance scoring |
| prp | Pairwise Ranking Prompting |

### Agentic Methods

| Method | Description |
|--------|-------------|
| react | ReAct (Reason + Act with tools) |
| decomp | Decomposed Prompting |

**Reference**: [doc/paper/baselines.md](../paper/baselines.md)

---

## Creating a New Method

1. **Single-file method**:
   ```python
   # methods/mymethod.py
   from .base import BaseMethod
   from utils.llm import llm_call

   class MyMethod(BaseMethod):
       name = "mymethod"

       def evaluate_ranking(self, query: str, context: str, k: int = 1) -> str:
           response = llm_call(prompt=f"...", system="...")
           return self._parse_ranking(response)
   ```

2. **Register in `__init__.py`**:
   ```python
   from .mymethod import MyMethod

   METHOD_REGISTRY = {
       ...
       "mymethod": (MyMethod, False),  # (class, supports_defense)
   }
   ```

3. **Add to `METHOD_CHOICES`** in `utils/arguments.py`:
   ```python
   METHOD_CHOICES = [..., "mymethod"]
   ```
