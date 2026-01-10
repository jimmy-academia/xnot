# SCALE: Structured Context Analysis & Logic Evaluation

**SCALE** is a benchmark designed to systematically identify the reasoning boundaries of Large Language Models (LLMs) by testing their ability to perform multi-hop logic over large, unstructured contexts.

## Core Philosophy

Unlike traditional "needle-in-a-haystack" benchmarks which test retrieval, **SCALE** tests **Synthesis**.
-   **No Simple Lookups**: All tasks require combining multiple pieces of information.
-   **Fail-Positive Design**: Tasks are only accepted if standard LLMs (like GPT-4o) initially fail or struggle, ensuring the benchmark measures the *cutting edge*.
-   **Reasoning Chains**: Success is measured not just by the final answer, but by the correctness of the intermediate "Reasoning Primitives" (Evidence).

## The 3-3-2-2 Taxonomy

SCALE organizes 100 tasks into 4 Real-World Perspectives to simulate actual user needs:

1.  **The Customer** (Decisions): Health, Social, Value.
2.  **The Owner** (Optimization): Talent, Efficiency, Strategy.
3.  **The Researcher** (Analysis): Psychology, Trends.
4.  **The Moderator** (Integrity): Forensics, Safety.

## Key Dimensions

| Dimension | Description |
| :--- | :--- |
| **Context** | Full review history of a restaurant (simulating "Too much to read"). |
| **Primitives (P)** | The intermediate facts/logic steps required to solve a Query. |
| **Verdict** | The final decision, strictly derived from P. |

## Goal
To move beyond "Vibe Checks" and towards **Reliable, Auditable Reasoning** over long contexts.
