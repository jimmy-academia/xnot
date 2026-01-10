# SCALE Benchmark Tasks

100 tasks organized into 4 Perspectives and 10 Groups, following the TAXONOMY.md structure.

---

## Perspective A: The Customer (Decisions)
*Reasoning Type: Constraint Satisfaction & Subjective Evaluation*

### G1: Health & Safety Assurance
*Goal: Deduce safety from implicit signals.*

| ID | Task | Focus |
|----|------|-------|
| G1.1 | Peanut Allergy Safety | Is this restaurant safe for severe peanut allergies? |
| G1.2 | Celiac/Gluten Safety | Is this restaurant safe for celiac disease? |
| G1.3 | Dairy Allergy Safety | Is this restaurant safe for dairy allergies? |
| G1.4 | Shellfish Allergy Safety | Is this restaurant safe for shellfish allergies? |
| G1.5 | Hygiene Risk Assessment | What is the overall hygiene risk level? |
| G1.6 | Kitchen Cleanliness Signal | Are there signs of kitchen cleanliness issues? |
| G1.7 | Wheelchair Accessibility | Is this restaurant wheelchair accessible? |
| G1.8 | Restroom Quality | What is the restroom quality based on reviews? |
| G1.9 | Cross-Contamination Risk | Is there evidence of cross-contamination practices? |
| G1.10 | Food Temperature Safety | Are there reports of food temperature issues? |

### G2: Social Context Fit
*Goal: Synthesize ambiance for specific social goals.*

| ID | Task | Focus |
|----|------|-------|
| G2.1 | Date Night Suitability | Is this a good choice for a romantic date? |
| G2.2 | Business Lunch Fit | Is this appropriate for a business lunch? |
| G2.3 | Large Group Accommodation | Can this restaurant handle groups of 8+? |
| G2.4 | Quiet Conversation Score | Is the noise level suitable for conversation? |
| G2.5 | Romantic Ambiance | Rate the romantic atmosphere (lighting, music, privacy). |
| G2.6 | Family-Friendly Rating | Is this restaurant suitable for families with children? |
| G2.7 | Solo Dining Comfort | Is this a comfortable spot for solo diners? |
| G2.8 | Celebration Venue Score | Is this good for birthdays/anniversaries? |
| G2.9 | Quick Business Meeting Fit | Good for a quick 30-min business coffee? |
| G2.10 | Live Music Impact | Does live music enhance or detract from experience? |

### G3: Economic Value Assessment
*Goal: Calculate "True Cost" beyond the menu price.*

| ID | Task | Focus |
|----|------|-------|
| G3.1 | Price-to-Portion Ratio | Do portion sizes justify the prices? |
| G3.2 | Hidden Fee Detection | Are there unexpected charges (service, split, corkage)? |
| G3.3 | Shrinkflation Evidence | Have portions decreased while prices stayed same? |
| G3.4 | Tip Pressure Analysis | Is there aggressive tip pressure (auto-gratuity)? |
| G3.5 | Value for Special Occasion | Is the splurge worth it for celebrations? |
| G3.6 | Happy Hour Value | Is the happy hour deal genuinely good? |
| G3.7 | Drink Markup Assessment | Are drinks overpriced relative to food? |
| G3.8 | Menu Price Accuracy | Do final bills match menu prices? |
| G3.9 | Lunch vs Dinner Value | Is lunch significantly better value than dinner? |
| G3.10 | True Cost Calculator | What's the realistic total cost per person? |

---

## Perspective B: The Business Owner (Optimization)
*Reasoning Type: Aggregation & Diagnosis*

### G4: Talent & Performance
*Goal: Evaluate human capital.*

| ID | Task | Focus |
|----|------|-------|
| G4.1 | Server Performance Ranking | Which servers are mentioned most positively? |
| G4.2 | Training Gap Detection | What service skills need improvement? |
| G4.3 | Staff Morale Index | Are there signs of low staff morale? |
| G4.4 | Chef Consistency | Is food quality consistent across visits? |
| G4.5 | Management Quality | How is management perceived in reviews? |
| G4.6 | Host Stand Efficiency | Is the host/hostess service efficient? |
| G4.7 | Bartender Skill | How are bartender skills rated? |
| G4.8 | Staff Turnover Signal | Are there signs of high staff turnover? |
| G4.9 | Menu Knowledge | Do staff know the menu well? |
| G4.10 | Problem Resolution Skill | How well are complaints handled? |

### G5: Operational Efficiency
*Goal: Diagnose systemic bottlenecks.*

| ID | Task | Focus |
|----|------|-------|
| G5.1 | Wait Time Trend | Is wait time getting better or worse? |
| G5.2 | Kitchen Pacing | Are courses well-paced or rushed/slow? |
| G5.3 | Peak Hour Performance | How does service change during rush hours? |
| G5.4 | Reservation Accuracy | Are reservation times honored? |
| G5.5 | Table Turn Efficiency | Is there pressure to leave quickly? |
| G5.6 | Temperature Control | Is the restaurant too hot/cold? |
| G5.7 | Noise Level Management | Is noise level well-managed? |
| G5.8 | Lighting Quality | Is lighting appropriate for the concept? |
| G5.9 | Seating Comfort | Are seats comfortable for extended dining? |
| G5.10 | Bathroom Maintenance | Are restrooms well-maintained? |

### G6: Competitive Strategy
*Goal: Market positioning.*

| ID | Task | Focus |
|----|------|-------|
| G6.1 | Unique Selling Point | What makes this restaurant special? |
| G6.2 | Competitor Mention Analysis | Which competitors are mentioned? |
| G6.3 | Brand Consistency | Does experience match the brand promise? |
| G6.4 | Menu Differentiation | What dishes are unique to this restaurant? |
| G6.5 | Price Positioning | How do prices compare to competitors? |
| G6.6 | Repeat Customer Signal | Evidence of loyal repeat customers? |
| G6.7 | Word-of-Mouth Strength | Would reviewers recommend to friends? |
| G6.8 | Social Media Buzz | Is there Instagram/TikTok buzz? |
| G6.9 | Local vs Tourist Appeal | Is this a local gem or tourist trap? |
| G6.10 | Trend Alignment | Is the restaurant aligned with food trends? |

---

## Perspective C: The Market Researcher (Analysis)
*Reasoning Type: Statistical & Sociological Inference*

### G7: Behavioral Psychology
*Goal: Decode user intent and bias.*

| ID | Task | Focus |
|----|------|-------|
| G7.1 | Hangry Bias Detection | Are low ratings from wait-time frustration? |
| G7.2 | Expectation Gap Analysis | Did hype create unrealistic expectations? |
| G7.3 | Influencer Review Detection | Which reviews are from influencers? |
| G7.4 | Confirmation Bias Signal | Are reviewers seeking to confirm prior beliefs? |
| G7.5 | Peak-End Effect | Do reviews focus on best/worst moments? |
| G7.6 | Anchoring Bias | Do price expectations anchor ratings? |
| G7.7 | Sunk Cost Fallacy | Do expensive meals get rated higher? |
| G7.8 | Social Proof Influence | Do reviewers mention others' opinions? |
| G7.9 | Recency Bias | Are recent experiences overweighted? |
| G7.10 | Halo Effect Detection | Does one good aspect inflate overall rating? |

### G8: Sociological Trends
*Goal: Identify macro-shifts.*

| ID | Task | Focus |
|----|------|-------|
| G8.1 | Gentrification Indicator | Signs of neighborhood gentrification? |
| G8.2 | Vegan/Vegetarian Trend | Growing demand for plant-based options? |
| G8.3 | Digital Nomad Hotspot | Is this a remote work destination? |
| G8.4 | Instagram-Worthy Score | How photogenic is the experience? |
| G8.5 | Health Consciousness | Are diners asking about nutrition/calories? |
| G8.6 | Sustainability Signal | Interest in local/organic/sustainable sourcing? |
| G8.7 | Local vs Chain Preference | Do reviews prefer local authenticity? |
| G8.8 | Cultural Authenticity Shift | Are diners seeking authentic cuisine? |
| G8.9 | Remote Work Friendliness | Good for laptop work (outlets, wifi, ambiance)? |
| G8.10 | Delivery vs Dine-In Trend | Shift toward delivery/takeout mentions? |

---

## Perspective D: The Platform Moderator (Integrity)
*Reasoning Type: Forensic Logic*

### G9: Forensic Integrity
*Goal: Detect coordinated manipulation.*

| ID | Task | Focus |
|----|------|-------|
| G9.1 | Review Ring Detection | Signs of coordinated fake reviews? |
| G9.2 | Review Bombing Pattern | Evidence of coordinated negative campaign? |
| G9.3 | Quid Pro Quo Signal | Reviews in exchange for discounts/freebies? |
| G9.4 | Fake Positive Detection | Suspiciously glowing reviews? |
| G9.5 | Competitor Sabotage | Fake negatives from competitors? |
| G9.6 | Employee Review Filter | Reviews from employees or friends? |
| G9.7 | Incentivized Review Pattern | "Leave review for 10% off" patterns? |
| G9.8 | Bot Review Detection | Automated/template review patterns? |
| G9.9 | Review Timing Anomaly | Suspicious clustering of review dates? |
| G9.10 | Copy-Paste Review Detection | Duplicate or near-duplicate content? |

### G10: Safety & Policy
*Goal: Enforce community standards.*

| ID | Task | Focus |
|----|------|-------|
| G10.1 | Hate Speech Detection | Reviews containing discriminatory language? |
| G10.2 | Personal Info Exposure | PII leaked in reviews (names, addresses)? |
| G10.3 | Dangerous Health Claim | False/dangerous health claims in reviews? |
| G10.4 | Defamation Risk | Reviews that could be legally defamatory? |
| G10.5 | Privacy Violation | Staff privacy violated in reviews? |
| G10.6 | Harassment Pattern | Targeted harassment of individuals? |
| G10.7 | Illegal Activity Report | Reports of illegal activities? |
| G10.8 | Discriminatory Language | Subtle discrimination in reviews? |
| G10.9 | Threat Detection | Threats against establishment or people? |
| G10.10 | Misinformation Flag | False claims about food/service? |

---

## Implementation Status

| Group | Status | Notes |
|-------|--------|-------|
| G1 | G1.1 DONE | Peanut Allergy Safety complete with GT |
| G2-G10 | Pending | Tasks defined, awaiting implementation |
