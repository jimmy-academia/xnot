# Research Reflection: ANoT Evaluation Framework

*Generated: January 2026*

## What This Project Accomplishes

This framework investigates a fundamental question in LLM research:

> **Do LLMs need architectural support (tools, selective access, parallel execution) to handle real-world constraint satisfaction over heterogeneous data at scale?**

The framework goes beyond "which prompt is best" to test whether **reasoning architecture** matters more than **prompting technique**.

### Core Hypothesis

Methods that parse query structure explicitly, access data selectively, and execute in parallel (ANoT) will outperform monolithic full-context approaches (CoT) as:
1. Logical complexity increases (G01 → G10)
2. Candidate pool scales (10 → 50)
3. Adversarial conditions intensify (13 attack types)

### Benchmark Design: Philadelphia Cafes

| Property | Design Choice | Why It Matters |
|----------|---------------|----------------|
| Ground truth | Machine-verifiable logical constraints | Eliminates human judgment bias |
| Complexity | 10 groups (G01-G10) with progressive difficulty | Isolates specific reasoning capabilities |
| Data sources | Structured (attributes) + Unstructured (reviews) | Tests multi-source reasoning |
| Adversarial | 13 attack types on non-gold items only | Tests robustness without corrupting evaluation |
| Scale | 10-50 candidates | Reveals degradation patterns |

---

## The Value of This Work

### Rare and Important Contributions

1. **Failure mode testing** - Most benchmarks test surface capabilities. This tests *how methods fail* under realistic conditions:
   - Token budget exhaustion
   - Logical structure complexity (nested AND/OR)
   - Adversarial manipulation (prompt injection, sarcasm)
   - Scale degradation

2. **Rigorous methodology** - Reproducible attacks, per-request seeding, position bias mitigation make results trustworthy.

3. **Architectural insight** - ANoT treats recommendation as **program synthesis** rather than text completion:
   - Phase 1: Parse query → logical structure
   - Phase 2: Build execution plan via ReAct
   - Phase 3: Execute DAG in parallel

4. **Practical implications** - Results inform real RAG system design: when to use monolithic prompts vs. agentic architectures.

---

## Blind Spots to Address

### 1. Testing Execution, Not Interpretation

Current requests have explicit logical structure:
```json
{"op": "AND", "args": [
  {"aspect": "drive_thru", "evidence": {...}},
  {"aspect": "good_for_kids", "evidence": {...}}
]}
```

But real users say: *"somewhere nice to work for a few hours"*

This requires **inferring** constraints (WiFi, quiet, seating) from fuzzy intent. The benchmark heavily tests Phase 3 (execution) but real-world value may depend on Phase 1's ability to interpret underspecified requests.

**Question to explore**: How does ANoT perform when the logical structure must be inferred, not given?

### 2. Sophistication as Fragility

ANoT's 3-phase architecture has multiple failure points:
- Phase 1 misparses "OR" as "AND" → wrong candidates
- Phase 2 prunes too aggressively → gold item dropped
- Phase 3 DAG has dependency bug → corrupted results

CoT's simplicity means fewer things can break.

**Question to explore**: What's the reliability rate of the ANoT pipeline itself, independent of task difficulty?

### 3. Benchmark-Architecture Alignment

The task design (logical constraints, structured paths, exactly 1 correct answer) aligns well with ANoT's approach. Different tasks (creative recommendations, preference-based ranking, subjective queries) might favor different architectures.

**Question to explore**: Does ANoT's advantage generalize to other constraint-satisfaction domains?

---

## Keys to Progress

### 1. Error Taxonomy Over Aggregate Metrics

Move from Hits@K/MRR to understanding *why* methods fail:

**Phase-level attribution** (ANoT):
- Phase 1: Logical structure misparsed
- Phase 2: Gold item incorrectly pruned
- Phase 3: Correct candidates, wrong final ranking

**Condition-level attribution** (all methods):
- Which constraint types are hardest?
- Which complexity groups break which methods?

**Failure mode clustering**:
- "Distracted by injection"
- "Missed evidence in long review"
- "Misunderstood negation"

### 2. Ablation Studies

Which component of ANoT contributes most?
- Skip Phase 2 (no ReAct pruning) - does it still outperform?
- Simplify Phase 1 (rule-based parser) - does LLM parsing matter?
- Remove parallelism (sequential execution) - how much does DAG help?

### 3. Hybrid Exploration

The gap between CoT (1 call, all context) and ANoT (50+ calls, selective access) suggests middle-ground approaches:
- Dict-mode CoT (selective access, single call)
- Simplified ANoT (2-phase, no ReAct)
- Listwise with selective context

### 4. Cross-Domain Validation

Test ANoT architecture on:
- Other recommendation domains (products, movies, jobs)
- Non-recommendation constraint satisfaction (scheduling, planning)
- Tasks with multiple valid answers

---

## Concrete Next Steps

### Immediate: Failure Analysis Pipeline

Create `failure_analysis.py` that for each incorrect prediction:
1. Identifies which condition(s) were evaluated wrongly
2. Traces back to the phase/step where error originated
3. Categorizes into taxonomy (built iteratively)

Output: "ANoT is better *because* it handles X, Y, Z while CoT fails on them"

### Medium-term: Ablation Suite

Systematic ablations to isolate contributions:
- `anot_no_prune`: Skip Phase 2 pruning
- `anot_sequential`: No parallel execution
- `anot_simple_parse`: Rule-based Phase 1
- `cot_dict`: CoT with selective data access

### Long-term: Generalization Study

Apply framework to:
- Different domains (same architecture, new data)
- Different task structures (multi-answer, ranked preferences)
- Real user queries (ambiguous intent, implicit constraints)

---

## Summary

This framework makes a valuable contribution by rigorously testing *when and why* architectural approaches outperform prompting techniques. The key insight to pursue:

> **Move from "ANoT achieves X% accuracy" to "ANoT succeeds because of mechanism M on condition type T" — that's the publishable, generalizable finding.**

The benchmark infrastructure is solid. The next phase is deeper analysis of failure modes and systematic ablation to isolate the sources of ANoT's advantages.
