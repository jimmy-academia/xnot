"""
Legacy helper functions - prefer using utils.experiment.ExperimentManager.
"""

import glob
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("results")


def get_next_run_number() -> int:
    """Scan results/ and find the next available run number."""
    existing = glob.glob(str(RESULTS_DIR / "[0-9]*_*/"))
    if not existing:
        return 1
    numbers = []
    for p in existing:
        folder_name = Path(p).name
        try:
            num = int(folder_name.split("_")[0])
            numbers.append(num)
        except ValueError:
            continue
    return max(numbers) + 1 if numbers else 1


def create_run_dir(run_name: str) -> Path:
    """Create a numbered run directory and return its path."""
    RESULTS_DIR.mkdir(exist_ok=True)
    run_num = get_next_run_number()
    run_dir = RESULTS_DIR / f"{run_num}_{run_name}"
    run_dir.mkdir(exist_ok=True)
    return run_dir
