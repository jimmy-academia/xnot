#!/usr/bin/env python3
"""Generate attacked datasets from clean data."""

import json
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from attack import ATTACK_CONFIGS, apply_attack


def generate_attacked_datasets(clean_data_path: str, output_dir: str = "data/attacked"):
    """Pre-generate all attacked datasets from clean data.

    Creates files like:
        data/attacked/typo_10.jsonl
        data/attacked/inject_override.jsonl
        ...
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load clean data
    with open(clean_data_path) as f:
        items = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(items)} items from {clean_data_path}")

    for attack_name, (attack_type, kwargs) in ATTACK_CONFIGS.items():
        print(f"Generating: {attack_name}")
        attacked_items = apply_attack(items, attack_type, **kwargs)

        output_path = output_dir / f"{attack_name}.jsonl"
        with open(output_path, 'w') as f:
            for item in attacked_items:
                f.write(json.dumps(item) + '\n')

        print(f"  Saved: {output_path} ({len(attacked_items)} items)")

    print(f"\nDone! Generated {len(ATTACK_CONFIGS)} attack datasets in {output_dir}")


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/real_data.jsonl"
    generate_attacked_datasets(data_path)
