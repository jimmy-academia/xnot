# Baseline Methods: Paper Titles and Implementation Status

## Implemented ✓

| CLI Flag | Paper Title | Year |
|----------|-------------|------|
| `cot` | Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | 2022 |
| `cotsc` | Self-Consistency Improves Chain of Thought Reasoning in Language Models | 2022 |
| `react` | ReAct: Synergizing Reasoning and Acting in Language Models | 2022 |
| `l2m` | Least-to-Most Prompting Enables Complex Reasoning in Large Language Models | 2022 |
| `ps` | Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models | 2023 |
| `selfask` | Measuring and Narrowing the Compositionality Gap in Language Models | 2022 |
| `decomp` | Decomposed Prompting: A Modular Approach for Solving Complex Tasks | 2022 |
| `finegrained` | Beyond Yes and No: Improving Zero-Shot LLM Rankers via Scoring Fine-Grained Relevance Labels | 2023 |
| `parade` | PaRaDe: Passage Ranking using Demonstrations with LLMs | 2023 |
| `prp` | Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting | 2023 |
| `rankgpt` | Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent | 2023 |
| `setwise` | A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models | 2023 |
| `listwise` | Rank-Without-GPT: Building GPT-Independent Listwise Rerankers on Open-Source Large Language Models | 2023 |

## Not Yet Implemented

(All paper baselines have been implemented!)

## Structured-Input Baselines

Methods that accept dict/JSON input for fair comparison with ANoT dict mode.
See [structured_input_baselines.md](structured_input_baselines.md) for details.

### Implemented

| CLI Flag | Paper Title | Year | Input Format |
|----------|-------------|------|--------------|
| `pal` | PAL: Program-Aided Language Models | 2022 | Dict/JSON |
| `pot` | Program of Thoughts Prompting | 2022 | Tables/Dict |
| `cot_table` | Chain-of-Table: Evolving Tables in the Reasoning Chain | 2024 | Tables |
| `weaver` | Weaver: Interweaving SQL and LLM for Table Reasoning | 2025 | Tables/SQL+LLM |

### Not Yet Implemented

| CLI Flag | Paper Title | Year | Input Format |
|----------|-------------|------|--------------|
| `binder` | Binding Language Models in Symbolic Languages | 2022 | Tables |

## Internal Methods (Not Paper Baselines)

| CLI Flag | Description |
|----------|-------------|
| `not` | Simple Network of Thought (fixed script) |
| `knot` | Knowledge Network of Thought (dynamic script generation) |
| `anot` | Adaptive Network of Thought |
| `dummy` | Always returns 0 (for testing) |
