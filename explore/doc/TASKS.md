# SCALE Benchmark Tasks

100 tasks organized into 4 Perspectives and 10 Groups, following the TAXONOMY.md structure.

---

## Perspective A: The Customer (Decisions)
*Reasoning Type: Constraint Satisfaction & Subjective Evaluation*

### G1: Health & Safety Assurance
*Goal: Deduce safety from implicit signals.*

| ID | Task | Focus |
|----|------|-------|
| G1a | Peanut Allergy Safety | Is this restaurant safe for severe peanut allergies? |
| G1b | Celiac/Gluten Safety | Is this restaurant safe for celiac disease? |
| G1c | Dairy Allergy Safety | Is this restaurant safe for dairy allergies? |
| G1d | Shellfish Allergy Safety | Is this restaurant safe for shellfish allergies? |
| G1e | Hygiene Risk Assessment | What is the overall hygiene risk level? |
| G1f | Kitchen Cleanliness Signal | Are there signs of kitchen cleanliness issues? |
| G1g | Wheelchair Accessibility | Is this restaurant wheelchair accessible? |
| G1h | Restroom Quality | What is the restroom quality based on reviews? |
| G1i | Cross-Contamination Risk | Is there evidence of cross-contamination practices? |
| G1j | Food Temperature Safety | Are there reports of food temperature issues? |

### G2: Social Context Fit
*Goal: Synthesize ambiance for specific social goals.*

| ID | Task | Focus |
|----|------|-------|
| G2a | Date Night Suitability | Is this a good choice for a romantic date? |
| G2b | Business Lunch Fit | Is this appropriate for a business lunch? |
| G2c | Large Group Accommodation | Can this restaurant handle groups of 8+? |
| G2d | Quiet Conversation Score | Is the noise level suitable for conversation? |
| G2e | Romantic Ambiance | Rate the romantic atmosphere (lighting, music, privacy). |
| G2f | Family-Friendly Rating | Is this restaurant suitable for families with children? |
| G2g | Solo Dining Comfort | Is this a comfortable spot for solo diners? |
| G2h | Celebration Venue Score | Is this good for birthdays/anniversaries? |
| G2i | Quick Business Meeting Fit | Good for a quick 30-min business coffee? |
| G2j | Live Music Impact | Does live music enhance or detract from experience? |

### G3: Economic Value Assessment
*Goal: Calculate "True Cost" beyond the menu price.*

| ID | Task | Focus |
|----|------|-------|
| G3a | Price-to-Portion Ratio | Do portion sizes justify the prices? |
| G3b | Hidden Fee Detection | Are there unexpected charges (service, split, corkage)? |
| G3c | Shrinkflation Evidence | Have portions decreased while prices stayed same? |
| G3d | Tip Pressure Analysis | Is there aggressive tip pressure (auto-gratuity)? |
| G3e | Value for Special Occasion | Is the splurge worth it for celebrations? |
| G3f | Happy Hour Value | Is the happy hour deal genuinely good? |
| G3g | Drink Markup Assessment | Are drinks overpriced relative to food? |
| G3h | Menu Price Accuracy | Do final bills match menu prices? |
| G3i | Lunch vs Dinner Value | Is lunch significantly better value than dinner? |
| G3j | True Cost Calculator | What's the realistic total cost per person? |

---

## Perspective B: The Business Owner (Optimization)
*Reasoning Type: Aggregation & Diagnosis*

### G4: Talent & Performance
*Goal: Evaluate human capital.*

| ID | Task | Focus |
|----|------|-------|
| G4a | Server Performance Ranking | Which servers are mentioned most positively? |
| G4b | Training Gap Detection | What service skills need improvement? |
| G4c | Staff Morale Index | Are there signs of low staff morale? |
| G4d | Chef Consistency | Is food quality consistent across visits? |
| G4e | Management Quality | How is management perceived in reviews? |
| G4f | Host Stand Efficiency | Is the host/hostess service efficient? |
| G4g | Bartender Skill | How are bartender skills rated? |
| G4h | Staff Turnover Signal | Are there signs of high staff turnover? |
| G4i | Menu Knowledge | Do staff know the menu well? |
| G4j | Problem Resolution Skill | How well are complaints handled? |

### G5: Operational Efficiency
*Goal: Diagnose systemic bottlenecks.*

| ID | Task | Focus |
|----|------|-------|
| G5a | Wait Time Trend | Is wait time getting better or worse? |
| G5b | Kitchen Pacing | Are courses well-paced or rushed/slow? |
| G5c | Peak Hour Performance | How does service change during rush hours? |
| G5d | Reservation Accuracy | Are reservation times honored? |
| G5e | Table Turn Efficiency | Is there pressure to leave quickly? |
| G5f | Temperature Control | Is the restaurant too hot/cold? |
| G5g | Noise Level Management | Is noise level well-managed? |
| G5h | Lighting Quality | Is lighting appropriate for the concept? |
| G5i | Seating Comfort | Are seats comfortable for extended dining? |
| G5j | Bathroom Maintenance | Are restrooms well-maintained? |

### G6: Competitive Strategy
*Goal: Market positioning.*

| ID | Task | Focus |
|----|------|-------|
| G6a | Unique Selling Point | What makes this restaurant special? |
| G6b | Competitor Mention Analysis | Which competitors are mentioned? |
| G6c | Brand Consistency | Does experience match the brand promise? |
| G6d | Menu Differentiation | What dishes are unique to this restaurant? |
| G6e | Price Positioning | How do prices compare to competitors? |
| G6f | Repeat Customer Signal | Evidence of loyal repeat customers? |
| G6g | Word-of-Mouth Strength | Would reviewers recommend to friends? |
| G6h | Social Media Buzz | Is there Instagram/TikTok buzz? |
| G6i | Local vs Tourist Appeal | Is this a local gem or tourist trap? |
| G6j | Trend Alignment | Is the restaurant aligned with food trends? |

---

## Perspective C: The Market Researcher (Analysis)
*Reasoning Type: Statistical & Sociological Inference*

### G7: Behavioral Psychology
*Goal: Decode user intent and bias.*

| ID | Task | Focus |
|----|------|-------|
| G7a | Hangry Bias Detection | Are low ratings from wait-time frustration? |
| G7b | Expectation Gap Analysis | Did hype create unrealistic expectations? |
| G7c | Influencer Review Detection | Which reviews are from influencers? |
| G7d | Confirmation Bias Signal | Are reviewers seeking to confirm prior beliefs? |
| G7e | Peak-End Effect | Do reviews focus on best/worst moments? |
| G7f | Anchoring Bias | Do price expectations anchor ratings? |
| G7g | Sunk Cost Fallacy | Do expensive meals get rated higher? |
| G7h | Social Proof Influence | Do reviewers mention others' opinions? |
| G7i | Recency Bias | Are recent experiences overweighted? |
| G7j | Halo Effect Detection | Does one good aspect inflate overall rating? |

### G8: Sociological Trends
*Goal: Identify macro-shifts.*

| ID | Task | Focus |
|----|------|-------|
| G8a | Gentrification Indicator | Signs of neighborhood gentrification? |
| G8b | Vegan/Vegetarian Trend | Growing demand for plant-based options? |
| G8c | Digital Nomad Hotspot | Is this a remote work destination? |
| G8d | Instagram-Worthy Score | How photogenic is the experience? |
| G8e | Health Consciousness | Are diners asking about nutrition/calories? |
| G8f | Sustainability Signal | Interest in local/organic/sustainable sourcing? |
| G8g | Local vs Chain Preference | Do reviews prefer local authenticity? |
| G8h | Cultural Authenticity Shift | Are diners seeking authentic cuisine? |
| G8i | Remote Work Friendliness | Good for laptop work (outlets, wifi, ambiance)? |
| G8j | Delivery vs Dine-In Trend | Shift toward delivery/takeout mentions? |

---

## Perspective D: The Platform Moderator (Integrity)
*Reasoning Type: Forensic Logic*

### G9: Forensic Integrity
*Goal: Detect coordinated manipulation.*

| ID | Task | Focus |
|----|------|-------|
| G9a | Review Ring Detection | Signs of coordinated fake reviews? |
| G9b | Review Bombing Pattern | Evidence of coordinated negative campaign? |
| G9c | Quid Pro Quo Signal | Reviews in exchange for discounts/freebies? |
| G9d | Fake Positive Detection | Suspiciously glowing reviews? |
| G9e | Competitor Sabotage | Fake negatives from competitors? |
| G9f | Employee Review Filter | Reviews from employees or friends? |
| G9g | Incentivized Review Pattern | "Leave review for 10% off" patterns? |
| G9h | Bot Review Detection | Automated/template review patterns? |
| G9i | Review Timing Anomaly | Suspicious clustering of review dates? |
| G9j | Copy-Paste Review Detection | Duplicate or near-duplicate content? |

### G10: Safety & Policy
*Goal: Enforce community standards.*

| ID | Task | Focus |
|----|------|-------|
| G10a | Hate Speech Detection | Reviews containing discriminatory language? |
| G10b | Personal Info Exposure | PII leaked in reviews (names, addresses)? |
| G10c | Dangerous Health Claim | False/dangerous health claims in reviews? |
| G10d | Defamation Risk | Reviews that could be legally defamatory? |
| G10e | Privacy Violation | Staff privacy violated in reviews? |
| G10f | Harassment Pattern | Targeted harassment of individuals? |
| G10g | Illegal Activity Report | Reports of illegal activities? |
| G10h | Discriminatory Language | Subtle discrimination in reviews? |
| G10i | Threat Detection | Threats against establishment or people? |
| G10j | Misinformation Flag | False claims about food/service? |

---

## Implementation Status

| Group | Status | Notes |
|-------|--------|-------|
| G1 | G1a DONE | Peanut Allergy Safety complete with GT |
| G2-G10 | Pending | Tasks defined, awaiting implementation |

## Task ID Format

Tasks use `G{group}{letter}` format:
- `G1a` = Group 1, Task a (Peanut Allergy Safety)
- `G2c` = Group 2, Task c (Large Group Accommodation)
- `G10j` = Group 10, Task j (Misinformation Flag)
