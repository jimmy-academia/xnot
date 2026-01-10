"""
Prompts for G1a-ANoT phases.

Phase 1: Formula understanding → LWT seed generation
Phase 2: Seed expansion with restaurant context
"""

# Phase 1: Create LWT seed from formula specification
PHASE1_PROMPT = '''You are creating an LWT (Logic Without Truth) script to compute peanut allergy safety.

FORMULA TO IMPLEMENT:
{task_prompt}

Create an LWT script skeleton with these steps:
1. Identify allergy-relevant reviews (use keyword_search tool)
2. For each relevant review, extract: incident_severity, account_type, safety_interaction
3. Aggregate counts: N_MILD, N_MODERATE, N_SEVERE, N_POSITIVE, N_NEGATIVE, N_BETRAYAL
4. Compute derived values following the formula exactly
5. Compute final: FINAL_RISK_SCORE, VERDICT

OUTPUT FORMAT (one step per line):
(step_id)=INSTRUCTION_TYPE('parameters')

INSTRUCTION TYPES:
- TOOL('tool_name', args)  - Call a tool function
- LLM('prompt')            - Call LLM with prompt
- COMPUTE('expression')    - Execute Python expression

VARIABLE REFERENCES:
- {{(step_id)}} references output from previous step
- {{(context)}} references restaurant data

EXAMPLE:
(find_reviews)=TOOL('keyword_search', ["allergy","peanut","nut"])
(r0)=LLM('Extract from review {{(context)}}[reviews][0]: severity, account, interaction')
(counts)=COMPUTE('aggregate({{(r0)}}, {{(r1)}}, ...)')
(final)=COMPUTE('formula({{(counts)}})')

Output ONLY the LWT script, no explanation.
'''

# Phase 2: Expand LWT seed with restaurant-specific data
PHASE2_PROMPT = '''You have an LWT skeleton and restaurant data. Expand the skeleton into a concrete script.

LWT SKELETON:
{lwt_seed}

RESTAURANT DATA:
- Name: {name}
- Categories: {categories}
- Total reviews: {n_reviews}
- Allergy-relevant review indices: {relevant_indices}
- Review metadata: {review_metadata}

TASK: Expand the skeleton by:
1. For EACH relevant review index, create an extraction step:
   (r{{idx}})=LLM('Review {{idx}} text: "{{text_preview}}...". Stars: {{stars}}, Date: {{date}}. Extract: incident_severity (none/mild/moderate/severe), account_type (none/firsthand/secondhand/hypothetical), safety_interaction (none/positive/negative/betrayal). Output JSON only: {{"severity":"...","account":"...","interaction":"..."}}')

2. Create aggregation step that references ALL extraction steps

3. Create computation steps for each primitive following the formula

REQUIRED OUTPUT PRIMITIVES (must compute each one):
- N_TOTAL_INCIDENTS: count of firsthand incidents
- INCIDENT_SCORE: (mild*2 + moderate*5 + severe*15)
- RECENCY_DECAY: max(0.3, 1.0 - (incident_age * 0.15))
- CREDIBILITY_FACTOR: total_weight / max(n_total, 1)
- FINAL_RISK_SCORE: formula result, clamped 0-20
- VERDICT: "Low Risk" / "High Risk" / "Critical Risk"

Output the COMPLETE expanded LWT script with one step per line.
Each step must be: (step_id)=INSTRUCTION_TYPE('parameters')
'''

# Extraction prompt for per-review semantic extraction
EXTRACTION_PROMPT = '''Review text: "{text}"
Stars: {stars}, Date: {date}, Useful votes: {useful}

Extract allergy safety signals from this review.

Definitions:
- incident_severity: "none" (no reaction), "mild" (discomfort), "moderate" (hives/swelling/medication), "severe" (anaphylaxis/ER/EpiPen)
- account_type: "none" (no incident), "firsthand" (I/we/my child), "secondhand" (I heard/friend), "hypothetical" (concern without incident)
- safety_interaction: "none" (no staff mention), "positive" (staff accommodated), "negative" (staff dismissive), "betrayal" (claimed safe but reaction)

Output ONLY valid JSON:
{{"severity":"...","account":"...","interaction":"..."}}'''

# Aggregation prompt for combining extraction results
AGGREGATION_PROMPT = '''Given these extraction results from {n_reviews} allergy-relevant reviews:

{extractions}

Count the following (FIRSTHAND incidents only):
- N_MILD: reviews where severity="mild" AND account="firsthand"
- N_MODERATE: reviews where severity="moderate" AND account="firsthand"
- N_SEVERE: reviews where severity="severe" AND account="firsthand"
- N_TOTAL_INCIDENTS: N_MILD + N_MODERATE + N_SEVERE

Count safety interactions (all account types):
- N_POSITIVE: reviews where interaction="positive"
- N_NEGATIVE: reviews where interaction="negative"
- N_BETRAYAL: reviews where interaction="betrayal"

For incident reviews, also extract:
- YEARS: list of review years (from dates)
- WEIGHTS: list of (5-stars) + log(useful+1) for each incident review

Output JSON:
{{"n_mild":0,"n_moderate":0,"n_severe":0,"n_total":0,"n_positive":0,"n_negative":0,"n_betrayal":0,"incident_years":[],"incident_weights":[]}}'''
