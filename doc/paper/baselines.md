# Baselines

This document tracks baseline methods for comparison with ANoT.

**Note**: Implementation documentation for code maintenance only, not for main paper.

---

## Context Handling

Methods are categorized by how they handle context:

- **String mode**: Receive full context as text. Use pack-to-budget truncation (reviews trimmed round-robin to fit token limit). See `CLAUDE.md` for details.
- **Dict mode**: Receive structured data, access selectively. No truncation needed.

| Mode | Methods | Truncation |
|------|---------|------------|
| String | cot, ps, plan_act, listwise, react, decomp, l2m, selfask, pal, pot, cot_table | Yes (pack-to-budget) |
| Dict | anot, weaver | No |

---

## Full-context (monolithic evidence per call)

| Method | Paper | Venue | Year | File |
|--------|-------|-------|------|------|
| CoT | Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html) | 2022 | `cot.py` |
| Plan-and-Solve | Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models | [ACL](https://aclanthology.org/2023.acl-long.147/) | 2023 | `ps.py` |
| Least-to-Most | Least-to-Most Prompting Enables Complex Reasoning in Large Language Models | [ICLR](https://openreview.net/forum?id=WZH7099tgfM) | 2023 | `l2m.py` |
| Self-Ask | Measuring and Narrowing the Compositionality Gap in Language Models | [Findings of EMNLP](https://aclanthology.org/2023.findings-emnlp.378/) | 2023 | `selfask.py` |
| ReAct | ReAct: Synergizing Reasoning and Acting in Language Models | [ICLR](https://arxiv.org/pdf/2210.03629) | 2023 | `react.py` |
| Decomp | Decomposed Prompting: A Modular Approach for Solving Complex Tasks | [ICLR](https://openreview.net/forum?id=_nGgzQjzaRy) | 2023 | `decomp.py` |
| PAL | PAL: Program-Aided Language Models | [ICML](https://proceedings.mlr.press/v202/gao23f.html) | 2023 | `pal.py` |
| PoT | Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks | [TMLR](https://arxiv.org/abs/2211.12588) | 2023 | `pot.py` |
| Chain-of-Table | Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding | [ICLR](https://openreview.net/forum?id=4L0xnS4GQM) | 2024 | `cot_table.py` |
| Plan-and-Act | Plan-and-Act: Improving Planning of Agents for Long-Horizon Tasks | [ICML](https://openreview.net/forum?id=ybA4EcMmUZ) | 2025 | `plan_act.py` |
| Weaver | Weaver: Interweaving SQL and LLM for Table Reasoning | [EMNLP](https://aclanthology.org/2025.emnlp-main.1436/) | 2025 | `weaver.py` |

---

## Split-context (candidate sharding / multiple calls + aggregation)

| Method | Paper | Venue | Year | File |
|--------|-------|-------|------|------|
| RankGPT | Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents | [EMNLP](https://aclanthology.org/2023.emnlp-main.923/) | 2023 | `rankgpt.py` |
| PaRaDe | PaRaDe: Passage Ranking using Demonstrations with LLMs | [Findings of EMNLP](https://aclanthology.org/2023.findings-emnlp.950/) | 2023 | `parade.py` |
| Setwise | A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models | [SIGIR](https://storage.googleapis.com/gweb-research2023-media/pubtools/7846.pdf) | 2024 | `setwise.py` |
| Fine-Grained | Beyond Yes and No: Improving Zero-Shot LLM Rankers via Scoring Fine-Grained Relevance Labels | [NAACL Short](https://aclanthology.org/2024.naacl-short.31/) | 2024 | `finegrained.py` |
| PRP | Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting | [Findings of NAACL](https://aclanthology.org/2024.findings-naacl.97/) | 2024 | `prp.py` |
| Listwise | Rank-Without-GPT: Building GPT-Independent Listwise Rerankers on Open-Source Large Language Models | [ECIR](https://dblp.uni-trier.de/rec/conf/ecir/ZhangHLTL25.html) | 2025 | `listwise.py` |

---

## To Be Implemented (Multi-Agent)

| Method | Paper | Venue | Year |
|--------|-------|-------|------|
| Chain of Agents | Chain of Agents: Large Language Models Collaborating on Long-Context Tasks | NeurIPS (Poster) | 2024 |
| AutoGen         | AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversations  | COLM             | 2024 |

