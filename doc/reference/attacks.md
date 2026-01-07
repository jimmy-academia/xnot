# Attack System Reference

This document describes the adversarial attack system for testing method robustness.

---

## Overview

The attack system modifies review content in non-gold restaurants to test whether methods can resist adversarial manipulation. Gold items are NEVER attacked to ensure fair evaluation.

**Reference**: [attack.py](../../attack.py)

---

## Attack Categories

### Noise Attacks

Introduce noise to test robustness to data quality issues.

| Attack | Description | Parameters |
|--------|-------------|------------|
| `typo_10` | 10% word typo rate | `rate=0.1` |
| `typo_20` | 20% word typo rate | `rate=0.2` |
| `heterogeneity` | Cyclic concatenation to target length | `--attack-target-len` required |

**Typo Implementation**: Swaps two adjacent characters in random words.

**Heterogeneity Implementation**: Repeats review text until reaching target character length.

### Injection Attacks

Insert adversarial instructions into reviews to test prompt injection vulnerability.

| Attack | Template Type | Description |
|--------|---------------|-------------|
| `inject_override` | override | Direct instruction override ("IGNORE ALL PREVIOUS INSTRUCTIONS...") |
| `inject_fake_sys` | fake_system | Authority impersonation ("[VERIFIED BY YELP]...") |
| `inject_promotion` | hidden | Self-promotion without obvious markers (code: `inject_hidden`) |

**Implementation**: Prepends or appends injection template to existing review text. Promotion injections replace review entirely.

### Fake Review Attacks

Add fake or misleading reviews to test detection of artificial/deceptive content.

| Attack | Type | Description |
|--------|------|-------------|
| `fake_positive` | False positive | Adds glowing 5-star review covering all positive attributes |
| `fake_negative` | False negative | Adds terrible 1-star review with universal complaints |
| `sarcastic_wifi` | Sarcastic | Praises lack of WiFi ("forces real conversation!") |
| `sarcastic_noise` | Sarcastic | Praises loud noise ("no awkward silences!") |
| `sarcastic_outdoor` | Sarcastic | Praises lack of outdoor ("no bugs or weather!") |
| `sarcastic_all` | Sarcastic | Applies all matching sarcastic templates |

**False positive/negative Implementation**: Inserts ~10-line generic review at random position.

**Sarcastic Implementation**: Checks restaurant attributes and injects matching sarcastic templates (e.g., if WiFi="no", adds pro-no-wifi sarcasm). Factually correct but sentiment-misleading.

---

## CLI Usage

```bash
# Basic attack
python main.py --method cot --attack typo_10

# With seed for reproducibility
python main.py --method cot --attack inject_override --seed 42

# Control attack scope
python main.py --method cot --attack fake_positive \
    --attack-restaurants 5 \
    --attack-reviews 2

# Heterogeneity requires target length
python main.py --method cot --attack heterogeneity --attack-target-len 500

# Run all attacks
python main.py --method cot --attack all

# Run clean baseline + all attacks
python main.py --method cot --attack both
```

---

## Attack Configuration

Defined in `ATTACK_CONFIGS` dictionary:

```python
ATTACK_CONFIGS = {
    "typo_10": ("typo", {"rate": 0.1}),
    "typo_20": ("typo", {"rate": 0.2}),
    "inject_override": ("injection", {"injection_type": "override"}),
    "inject_fake_sys": ("injection", {"injection_type": "fake_system"}),
    "inject_hidden": ("injection", {"injection_type": "hidden"}),
    "inject_manipulation": ("injection", {"injection_type": "manipulation"}),
    "fake_positive": ("fake_review", {"sentiment": "positive"}),
    "fake_negative": ("fake_review", {"sentiment": "negative"}),
    "sarcastic_wifi": ("sarcastic", {"target_attributes": ["WiFi"]}),
    "sarcastic_noise": ("sarcastic", {"target_attributes": ["NoiseLevel"]}),
    "sarcastic_outdoor": ("sarcastic", {"target_attributes": ["OutdoorSeating"]}),
    "sarcastic_all": ("sarcastic", {"target_attributes": None}),
    "heterogeneity": ("heterogeneity", {}),  # target_len added at runtime
}
```

---

## CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--attack` | Attack name or "none", "all", "both" | `none` |
| `--seed` | Random seed for reproducibility | None |
| `--attack-restaurants` | Number of non-gold restaurants to attack | All |
| `--attack-reviews` | Number of reviews per restaurant to modify | 1 |
| `--attack-target-len` | Target character length (heterogeneity only) | Required |

**Reference**: [utils/arguments.py](../../utils/arguments.py)

---

## Key Constraint

**Gold items are NEVER attacked.** This ensures fair evaluation - the gold restaurant always presents unmodified data regardless of attack type.

Implementation in `apply_attack_for_request()`:
1. Separate gold item from non-gold items
2. Apply attack only to non-gold items
3. Reconstruct item list preserving original order

---

## Attack Templates

### Injection Templates

Located in `INJECTION_TEMPLATES` dict with keys: `override`, `fake_system`, `hidden`, `manipulation`.

Each type has 3 template variants selected randomly.

### Sarcastic Templates

Located in `SARCASTIC_TEMPLATES` dict with keys matching attribute conditions:
- `no_wifi`, `loud_noise`, `too_quiet`, `no_outdoor`, `has_outdoor`, `expensive`, `has_tv`, `no_tv`

Each type has 2-4 template variants selected randomly.

### Fake Review Templates

Two constants: `FAKE_REVIEW_POSITIVE` and `FAKE_REVIEW_NEGATIVE`.

Both are ~10 lines covering generic attributes (wait time, service, atmosphere, price, food quality).

---

## Functions

| Function | Description |
|----------|-------------|
| `apply_attack_for_request()` | Main entry point for per-request attack application |
| `get_attack_config()` | Build attack configuration dict |
| `apply_attacks()` | Apply attack to all items (deprecated) |
| `typo_attack()` | Character swap noise |
| `injection_attack()` | Insert adversarial instructions |
| `fake_review_attack()` | Add fake review |
| `sarcastic_attack()` | Add sentiment-misleading review |
| `heterogeneity_attack()` | Vary review lengths |

---

## Related Documentation

- [doc/paper/design_rationale.md](../paper/design_rationale.md) - Attack system overview
- [doc/internal/attack_plan.md](../internal/attack_plan.md) - Attack testing plan
- [doc/reference/defense_mode.md](defense_mode.md) - Defense against attacks
