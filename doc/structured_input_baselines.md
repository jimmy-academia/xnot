# Structured-Input Baseline Methods

Methods that can accept structured data (dict/JSON) as input, enabling fair comparison with ANoT's dict mode.

## Summary Table

| Method | Paper | Year | Input Format | Code Execution | Venue |
|--------|-------|------|--------------|----------------|-------|
| PAL | Program-Aided Language Models | 2022 | Dict/JSON | Python interpreter | ICML 2023 |
| PoT | Program of Thoughts Prompting | 2022 | Tables/Dict | Python interpreter | TMLR 2023 |
| Binder | Binding Language Models in Symbolic Languages | 2022 | Tables | SQL/Python + LM calls | ICLR 2023 |
| Chain-of-Table | Evolving Tables in the Reasoning Chain | 2024 | Tables | Table operations | ICLR 2024 |
| Weaver | Interweaving SQL and LLM for Table Reasoning | 2025 | Tables | SQL (pandas) + LLM | EMNLP 2025 |

---

## 1. PAL (Program-Aided Language Models)

**Paper**: "PAL: Program-Aided Language Models"
**ArXiv**: https://arxiv.org/abs/2211.10435
**GitHub**: https://github.com/reasoning-machines/pal
**Venue**: ICML 2023

### Key Idea
LLM generates Python code with reasoning steps; computation offloaded to Python interpreter.

### How It Handles Structured Input
- Native dict/JSON support via Python code generation
- Example: Object Counting task converts input to dict where keys=entities, values=quantities
- Generated code directly accesses dict keys: `data['attribute']`, `data['reviews'][0]`

### Adaptation for Restaurant Recommendation
```python
# Input provided as Python dict
restaurant = {
    "attributes": {"NoiseLevel": "quiet", "WiFi": "free"},
    "hours": {"Monday": "9:00-22:00"},
    "item_data": [{"review": "Great!", "rating": 5}]
}

# LLM generates code like:
def evaluate(restaurant, user_request):
    noise = restaurant['attributes'].get('NoiseLevel', 'unknown')
    is_quiet = noise in ['quiet', 'average']

    positive_reviews = sum(1 for r in restaurant['item_data']
                          if r['rating'] >= 4)

    if is_quiet and positive_reviews > 3:
        return 1  # recommend
    return 0  # neutral
```

### Key Insight
"The role of text is important - informative variable names are crucial for performance"

---

## 2. PoT (Program-of-Thoughts)

**Paper**: "Program of Thoughts Prompting: Disentangling Computation from Reasoning"
**ArXiv**: https://arxiv.org/abs/2211.12588
**GitHub**: https://github.com/TIGER-AI-Lab/Program-of-Thoughts
**Venue**: TMLR 2023

### Key Idea
Similar to PAL but emphasizes "binding semantic meanings to variables" and separating reasoning from computation.

### How It Handles Structured Input
- Supports table-based datasets (TabMWP, TATQA)
- Generated Python can parse and manipulate structured inputs
- Focus on multi-step reasoning with intermediate variable assignments

### Difference from PAL
- More emphasis on step-by-step "thought" decomposition
- Variables carry semantic meaning throughout computation
- Better suited for financial/numerical reasoning

### Adaptation for Restaurant Recommendation
```python
# Step 1: Extract relevant attributes
noise_level = restaurant['attributes']['NoiseLevel']  # semantic binding

# Step 2: Analyze reviews for user preferences
budget_mentions = [r for r in restaurant['item_data']
                   if 'cheap' in r['review'].lower() or 'affordable' in r['review'].lower()]

# Step 3: Combine evidence
meets_noise_req = noise_level == 'quiet'
is_budget_friendly = len(budget_mentions) >= 2

# Step 4: Final decision
recommendation = 1 if meets_noise_req and is_budget_friendly else 0
```

### Performance
~12% improvement over CoT across math/financial datasets; ~20% on FinQA (large numbers)

---

## 3. Binder

**Paper**: "Binding Language Models in Symbolic Languages"
**ArXiv**: https://arxiv.org/abs/2210.02875
**GitHub**: https://github.com/HKUNLP/Binder
**Project Page**: https://lm-code-binder.github.io/
**Venue**: ICLR 2023

### Key Idea
Training-free framework that binds LM API calls into SQL/Python code for hybrid symbolic-neural reasoning.

### How It Handles Structured Input
- Native table support with schema definitions
- Two special operators:
  - `QA("map@...")`: Maps messy values to clean SQL-compatible values
  - `QA("ans@...")`: Delegates complex reasoning to LM within query

### Example Binder Program
```sql
-- For question: "Is this restaurant good for quiet dining?"
SELECT QA("ans@is this good for quiet dining based on reviews?";
          SELECT review FROM item_data WHERE rating >= 4)
FROM restaurant
WHERE QA("map@noise level category"; NoiseLevel) = 'quiet'
```

### Adaptation for Restaurant Recommendation
```python
# Binder-style with LM API calls
def evaluate(restaurant):
    # Direct SQL-like access
    noise = restaurant['attributes']['NoiseLevel']

    # LM call for semantic matching
    noise_ok = QA("map@quiet environment?", noise)  # returns True/False

    # LM call for review analysis
    reviews = [r['review'] for r in restaurant['item_data']]
    sentiment = QA("ans@overall sentiment for budget dining?", reviews)

    return 1 if noise_ok and sentiment == 'positive' else 0
```

### Key Advantage
Combines precision of SQL with flexibility of LM reasoning; achieves SOTA on WikiTQ, TabFact with only ~12 examples.

---

## 4. Chain-of-Table

**Paper**: "Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding"
**ArXiv**: https://arxiv.org/abs/2401.04398
**Venue**: ICLR 2024

### Key Idea
Uses tables as intermediate "thoughts" - LLM iteratively applies table operations (filter, group, add_column) to reason.

### How It Handles Structured Input
- Native tabular format required
- Operations: add_column, select_row, select_column, group_by, sort_by
- Table evolves through reasoning chain

### Example Chain-of-Table Reasoning
```
Initial Table: restaurant attributes + reviews

Step 1: f_add_column("noise_suitable", NoiseLevel == 'quiet')
Step 2: f_select_row(rating >= 4)
Step 3: f_add_column("mentions_budget", contains(review, 'affordable'))
Step 4: f_group_by("noise_suitable").count()

Final: Answer based on transformed table
```

### Suitability for Restaurant Task
**Less suitable** because:
- Designed for tabular QA, not recommendation
- Requires data in strict table format
- Operations are predefined (not flexible like PAL/PoT)
- Better for fact verification than preference matching

---

## 5. Weaver

**Paper**: "Weaver: Interweaving SQL and LLM for Table Reasoning"
**ArXiv**: https://arxiv.org/abs/2505.18961
**GitHub**: https://github.com/CoRAL-ASU/weaver
**Venue**: EMNLP 2025

### Key Idea
Dynamically combines SQL for structured operations with LLM for semantic processing.
Generates step-by-step plans that interleave both operation types.

### How It Handles Structured Input
- Native table support with SQL-like operations
- Two-phase execution:
  - **SQL steps**: Filter, aggregate, sort using pandas (MySQL in original)
  - **LLM steps**: Semantic analysis, classification, reasoning on columns
- Dynamic plan generation based on question type

### Example Weaver Plan
```
Step 1: SQL: Filter reviews where stars >= 4
Step 2: LLM: Analyze review text for mentions of "quiet" -> quietness_score
Step 3: SQL: Count rows where quietness_score == "positive"
Final: Answer based on positive quiet mentions
```

### Adaptation for Restaurant Recommendation
```python
# Weaver generates a plan like:
# Step 1: SQL: Select high-rated reviews (stars >= 4)
# Step 2: LLM: Check if reviews mention user's preference -> relevant
# Step 3: SQL: Count positive matches
# Final: Recommend if enough positive evidence

def execute_plan(df, user_request):
    # Step 1: SQL
    df = df[df['stars'] >= 4]

    # Step 2: LLM (semantic)
    df['relevant'] = llm_check(df['review'], user_request)

    # Step 3: SQL
    positive_count = len(df[df['relevant'] == 'yes'])

    return 1 if positive_count >= 3 else 0
```

### Implementation Notes
- Original requires MySQL; our implementation uses pandas for portability
- Simplified plan format: "SQL:" and "LLM:" prefixes
- Safe eval for pandas queries with limited namespace
- Batch LLM processing for efficiency

---

## Recommendation for Implementation

### Priority Order
1. **PAL** (Highest) - Most direct dict support, simple to implement
2. **PoT** - Similar to PAL, good for comparison
3. **Weaver** - Most recent (2025), SQL+LLM hybrid approach
4. **Binder** - More complex but interesting hybrid approach
5. **Chain-of-Table** (Lowest) - Less suitable for recommendation task

### Implementation Notes
- All methods require Python code execution capability
- PAL/PoT: Just need to prompt LLM to generate Python that accesses dict
- Binder: Need to implement `QA()` wrapper for LM calls within generated code
- Chain-of-Table: Would need to restructure data as tables

---

## References

1. Gao et al. "PAL: Program-Aided Language Models" ICML 2023
2. Chen et al. "Program of Thoughts Prompting" TMLR 2023
3. Cheng et al. "Binding Language Models in Symbolic Languages" ICLR 2023
4. Wang et al. "Chain-of-Table" ICLR 2024
