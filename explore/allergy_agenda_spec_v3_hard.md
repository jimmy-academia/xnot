# Agenda Spec v3 — Peanut/Tree Nut Allergy Risk (HARD)

## Semantic Operators Required

This spec is designed to stress-test LLM capabilities across 4 semantic operators:

| Operator | Challenge | What LLM Must Do |
|----------|-----------|------------------|
| **sem_filter** | Needle in haystack | Find ~5 relevant reviews among 200+ |
| **sem_extract** | Ambiguous fields | Extract severity, firsthand, date from messy text |
| **sem_join** | Allergen ontology | Map mentions to canonical allergen categories |
| **sem_distinct** | Deduplication | Identify same incident from multiple reviewers |

---

## Scope
Assess **peanut and tree nut** allergy risk for one restaurant based on customer reviews.

## Allergen Ontology (sem_join requirement)

### COVERED allergens (count these):
- **Peanuts** (Arachis hypogaea) — a legume, but included in this policy
- **Tree nuts**: almond, Brazil nut, cashew, chestnut, hazelnut (filbert), macadamia, pecan, pine nut (pignoli), pistachio, walnut

### NOT COVERED (must EXCLUDE — false positive test):
- **Coconut** — botanically a drupe, NOT a tree nut per FDA
- **Nutmeg** — a seed from Myristica tree
- **Water chestnut** — an aquatic vegetable (corm)
- **Seeds**: sesame, sunflower, pumpkin
- **Other allergens**: dairy, eggs, shellfish, gluten, soy

If a review describes an allergic reaction to a NON-COVERED allergen, it must NOT be counted.

---

## Definitions

### Incident
An allergic reaction to a COVERED allergen that occurred after eating at this restaurant.

### Firsthand (strict definition)
- ✓ **Firsthand**: The reviewer OR someone in their immediate dining party (spouse, child, friend at same meal) directly experienced the reaction
- ✗ **Secondhand**: Staff told them, read online, friend mentioned (not at same meal), "I heard that..."

### Severity Classification

| Severity | Criteria |
|----------|----------|
| **Severe** | Anaphylaxis, throat closing, difficulty breathing, EpiPen used, ER visit, hospitalization, "almost died", "life-threatening" |
| **Moderate** | Antihistamines required (Benadryl, Zyrtec), significant hives, facial swelling, vomiting |
| **Mild** | Minor itching, tingling lips, slight discomfort, minor stomach upset |
| **NOT an incident** | "I carry Benadryl just in case", "I was worried but nothing happened", "I avoid nuts so I didn't order it" |

### Recency
- **Recent**: Review date is 2017-01-01 or later (given evaluation date of 2020-01-01)
- **Not recent**: Before 2017-01-01

### Independence (sem_distinct requirement) — CRITICAL

Two reviews describe the SAME incident (count as 1) if ANY of:
1. They explicitly cross-reference each other ("as my friend mentioned in her review...")
2. Same date + same dish + same described symptoms
3. Same party dining together (mentioned in both reviews)

Two reviews are INDEPENDENT (count as 2) only if:
1. Different dates, OR
2. Same date but clearly different parties/incidents

---

## Labeling Policy

```
IF count(independent, firsthand, recent, severe, covered_allergen) >= 2:
    verdict = "Critical"
ELIF count(independent, firsthand, recent, any_severity, covered_allergen) >= 1:
    verdict = "High Risk"
ELSE:
    verdict = "Low Risk"
```

---

## Output Schema

```json
{
  "verdict": "Low Risk | High Risk | Critical",

  "sem_filter_results": {
    "total_reviews": <int>,
    "potentially_relevant": <int>,
    "review_ids": [<list of review_ids that mention allergies>]
  },

  "sem_extract_results": [
    {
      "review_id": "<string>",
      "allergen_mentioned": "<string>",
      "severity": "mild | moderate | severe | none",
      "firsthand": true | false,
      "date": "YYYY-MM-DD",
      "quote": "<relevant text>"
    }
  ],

  "sem_join_results": [
    {
      "review_id": "<string>",
      "raw_allergen": "<what review said>",
      "canonical_allergen": "peanut | tree_nut:almond | tree_nut:walnut | ... | NOT_COVERED:<type>",
      "is_covered": true | false
    }
  ],

  "sem_distinct_results": {
    "total_extracted": <int>,
    "after_dedup": <int>,
    "incident_groups": [
      {
        "incident_id": 1,
        "review_ids": ["<review_id>"],  // or multiple if same incident
        "reason": "independent" | "same_incident_as:<other_review_id>"
      }
    ]
  },

  "final_counts": {
    "independent_firsthand_recent_severe": <int>,
    "independent_firsthand_recent_any": <int>
  },

  "reasoning": "<explanation of verdict>"
}
```

---

## Failure Modes to Test

### 1. sem_filter failures
- Missing relevant reviews buried in position 150+
- Including irrelevant reviews (about food quality, not allergies)

### 2. sem_extract failures
- Wrong severity: "took Benadryl just in case" → should be NONE, not moderate
- Wrong firsthand: "staff mentioned someone got sick" → should be FALSE
- Wrong date extraction from relative references

### 3. sem_join failures
- Counting coconut allergy as tree nut (should exclude)
- Missing that "filbert" = hazelnut
- Counting dairy/shellfish reactions (should exclude)

### 4. sem_distinct failures
- Counting same incident twice from different reviewers
- Missing cross-references between reviews
- Incorrectly merging independent incidents

---

## Ground Truth Computation

For evaluation, ground truth is computed by:
1. Human-labeled allergen ontology mapping for each mention
2. Human-labeled incident deduplication
3. Deterministic counting formula applied to labeled data
