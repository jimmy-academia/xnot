"""
Experiment management for LLM evaluation framework.
Handles directory creation, logging setup, and result saving.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

# Base results directory
RESULTS_DIR = Path("results")
DEV_DIR = RESULTS_DIR / "dev"
BENCHMARK_DIR = RESULTS_DIR / "benchmarks"


class ExperimentError(Exception):
    """Custom exception for experiment management errors."""
    pass


class ExperimentManager:
    """
    Unified experiment manager that handles:
    - Directory creation based on mode (dev vs benchmark)
    - Config and results saving
    """

    def __init__(self, run_name: str, benchmark_mode: bool = False):
        """
        Initialize experiment manager.

        Args:
            run_name: Name for this run
            benchmark_mode: If True, use benchmark directory (tracked in git)

        Raises:
            ExperimentError: If benchmark directory exists
        """
        self.run_name = run_name
        self.benchmark_mode = benchmark_mode
        self.run_dir: Optional[Path] = None
        self.config: Dict[str, Any] = {}
        self._created = False

    def setup(self) -> Path:
        """
        Create run directory and initialize.

        Returns:
            Path to the run directory

        Raises:
            ExperimentError: If benchmark directory exists and force=False
        """
        if self._created:
            return self.run_dir

        if self.benchmark_mode:
            self.run_dir = self._setup_benchmark_dir()
        else:
            self.run_dir = self._setup_dev_dir()

        self._created = True
        return self.run_dir

    def _setup_dev_dir(self) -> Path:
        """
        Create auto-numbered development directory.

        Pattern: results/dev/{NNN}_{run_name}/

        Reuses the last directory if it's empty.
        """
        DEV_DIR.mkdir(parents=True, exist_ok=True)

        # Check if last directory is empty - reuse it if so
        last_dir = self._get_last_dev_dir()
        if last_dir and self._is_dir_empty(last_dir):
            return last_dir

        run_num = self._get_next_run_number()
        dir_name = f"{run_num:03d}_{self.run_name}"
        run_dir = DEV_DIR / dir_name
        run_dir.mkdir(exist_ok=True)

        return run_dir

    def _get_last_dev_dir(self) -> Optional[Path]:
        """Get the most recent dev directory by number."""
        if not DEV_DIR.exists():
            return None

        existing = list(DEV_DIR.glob("[0-9][0-9][0-9]_*/"))
        if not existing:
            return None

        # Sort by run number descending
        def get_num(p):
            try:
                return int(p.name.split("_")[0])
            except ValueError:
                return -1

        existing.sort(key=get_num, reverse=True)
        return existing[0] if existing else None

    def _is_dir_empty(self, path: Path) -> bool:
        """Check if directory is empty (no files, subdirs ok if also empty)."""
        for item in path.iterdir():
            if item.is_file():
                return False
            if item.is_dir() and not self._is_dir_empty(item):
                return False
        return True

    def _setup_benchmark_dir(self) -> Path:
        """
        Create benchmark directory (no numbering).

        Pattern: results/benchmarks/{run_name}/

        Raises:
            ExperimentError: If directory already exists
        """
        BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

        run_dir = BENCHMARK_DIR / self.run_name

        if run_dir.exists():
            raise ExperimentError(
                f"Benchmark already exists: {run_dir}\n"
                f"Delete it manually or choose different --run-name"
            )

        run_dir.mkdir(exist_ok=True)
        return run_dir

    def _get_next_run_number(self) -> int:
        """Scan dev directory and find next available run number."""
        if not DEV_DIR.exists():
            return 1

        existing = list(DEV_DIR.glob("[0-9][0-9][0-9]_*/"))
        if not existing:
            return 1

        numbers = []
        for p in existing:
            folder_name = p.name
            try:
                num = int(folder_name.split("_")[0])
                numbers.append(num)
            except ValueError:
                continue

        return max(numbers) + 1 if numbers else 1

    def save_config(self, config: Dict[str, Any]) -> Path:
        """
        Save run configuration to config.json.

        Args:
            config: Configuration dictionary

        Returns:
            Path to saved config file
        """
        if not self._created:
            raise ExperimentError("Must call setup() before saving config")

        self.config = config.copy()
        self.config["timestamp"] = datetime.now().isoformat()
        self.config["benchmark_mode"] = self.benchmark_mode
        self.config["run_name"] = self.run_name

        config_path = self.run_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        return config_path

    def save_results(self, results: List[Dict], filename: str = "results.jsonl") -> Path:
        """
        Save evaluation results to JSONL file.

        Args:
            results: List of result dictionaries
            filename: Output filename (default: results.jsonl)

        Returns:
            Path to saved results file
        """
        if not self._created:
            raise ExperimentError("Must call setup() before saving results")

        result_path = self.run_dir / filename
        with open(result_path, 'w') as f:
            for r in sorted(results, key=lambda x: (x.get("item_id", ""), x.get("request_id", ""))):
                f.write(json.dumps(r) + '\n')

        return result_path

    def get_debug_logger(self, item_id: str, request_id: str):
        """
        Get a DebugLogger instance for a specific (item, request) pair.

        Args:
            item_id: Item identifier
            request_id: Request identifier

        Returns:
            DebugLogger instance
        """
        if not self._created:
            raise ExperimentError("Must call setup() before getting debug logger")

        from utils.logger import DebugLogger
        return DebugLogger(str(self.run_dir), item_id, request_id)

    def consolidate_debug_logs(self):
        """Consolidate debug logs after run completion."""
        if not self._created:
            return

        from utils.logger import consolidate_logs
        consolidate_logs(str(self.run_dir))

    @property
    def mode_str(self) -> str:
        """Human-readable mode string."""
        return "benchmark" if self.benchmark_mode else "development"

    def __str__(self) -> str:
        status = "created" if self._created else "not created"
        return f"ExperimentManager({self.run_name}, mode={self.mode_str}, {status})"

    def __repr__(self) -> str:
        return self.__str__()


def create_experiment(args) -> ExperimentManager:
    """
    Factory function to create ExperimentManager from parsed arguments.

    Args:
        args: Parsed argparse namespace with run_name, benchmark

    Returns:
        Configured ExperimentManager instance (not yet setup)
    """
    run_name = args.run_name or args.method or "unnamed"
    return ExperimentManager(
        run_name=run_name,
        benchmark_mode=getattr(args, 'benchmark', False)
    )
