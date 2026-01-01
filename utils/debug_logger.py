"""Debug logger for experiment debugging with buffered writes and interrupt handling."""

import json
import os
import signal
from datetime import datetime
from typing import Optional


class DebugLogger:
    """
    Logger that buffers entries in memory and flushes to disk at phase boundaries.
    Handles interrupts gracefully to preserve debug info.
    """

    def __init__(self, run_dir: str, item_id: str, request_id: str):
        """
        Initialize logger for a specific (item, request) pair.

        Args:
            run_dir: Path to the run directory (e.g., results/31_knot_v4)
            item_id: Unique identifier for the data item
            request_id: Request identifier (e.g., C0, C1, ...)
        """
        self.buffer = []
        self.run_dir = run_dir
        self.item_id = item_id
        self.request_id = request_id
        self.file_path = f"{run_dir}/temp_debug/{item_id}_{request_id}.jsonl"

        # Create directory if needed
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        # Store original handlers
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)

    def __enter__(self):
        """Context manager entry - register interrupt handlers."""
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - flush and restore handlers."""
        self.flush()
        signal.signal(signal.SIGINT, self._original_sigint)
        signal.signal(signal.SIGTERM, self._original_sigterm)
        return False

    def log(self, phase: str, event: str, data: Optional[dict] = None):
        """
        Add a log entry to the buffer.

        Args:
            phase: Phase identifier (e.g., "1.i", "2.b")
            event: Event type (e.g., "start", "end", "llm_call", "check", "fix")
            data: Optional dictionary with additional data
        """
        entry = {
            "ts": datetime.now().isoformat(),
            "item_id": self.item_id,
            "request_id": self.request_id,
            "phase": phase,
            "event": event,
            "data": data or {}
        }
        self.buffer.append(entry)

    def log_llm_call(self, phase: str, prompt: str, response: str):
        """Log an LLM call with prompt and response."""
        self.log(phase, "llm_call", {
            "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
            "response": response[:500] + "..." if len(response) > 500 else response,
            "prompt_len": len(prompt),
            "response_len": len(response)
        })

    def log_check(self, phase: str, passed: bool, reason: str = ""):
        """Log a check result."""
        self.log(phase, "check", {"passed": passed, "reason": reason})

    def log_fix(self, phase: str, action: str, details: Optional[dict] = None):
        """Log a fix action."""
        self.log(phase, "fix", {"action": action, **(details or {})})

    def flush(self):
        """Write buffered entries to disk and clear buffer."""
        if not self.buffer:
            return

        with open(self.file_path, "a") as f:
            for entry in self.buffer:
                f.write(json.dumps(entry) + "\n")
        self.buffer = []

    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signal - flush before exiting."""
        self.flush()
        # Restore original handler and re-raise
        signal.signal(signum, self._original_sigint if signum == signal.SIGINT else self._original_sigterm)
        raise KeyboardInterrupt


def consolidate_logs(run_dir: str):
    """
    Consolidate temp_debug files into organized debug directory.

    Args:
        run_dir: Path to the run directory

    Creates:
        - debug/by_item/{item_id}_{request_id}.jsonl
        - debug/by_phase/stage{N}_{phase}.jsonl
        - debug/summary.json
    """
    temp_dir = f"{run_dir}/temp_debug"
    debug_dir = f"{run_dir}/debug"

    if not os.path.exists(temp_dir):
        return

    # Create output directories
    os.makedirs(f"{debug_dir}/by_item", exist_ok=True)
    os.makedirs(f"{debug_dir}/by_phase", exist_ok=True)

    # Collect all entries
    all_entries = []
    phase_entries = {}

    for filename in os.listdir(temp_dir):
        if not filename.endswith(".jsonl"):
            continue

        filepath = os.path.join(temp_dir, filename)
        item_entries = []

        with open(filepath, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                item_entries.append(entry)
                all_entries.append(entry)

                # Group by phase
                phase = entry.get("phase", "unknown")
                if phase not in phase_entries:
                    phase_entries[phase] = []
                phase_entries[phase].append(entry)

        # Copy to by_item
        by_item_path = f"{debug_dir}/by_item/{filename}"
        with open(by_item_path, "w") as f:
            for entry in item_entries:
                f.write(json.dumps(entry) + "\n")

    # Write by_phase files
    for phase, entries in phase_entries.items():
        phase_filename = phase.replace(".", "_")
        by_phase_path = f"{debug_dir}/by_phase/{phase_filename}.jsonl"
        with open(by_phase_path, "w") as f:
            for entry in sorted(entries, key=lambda e: e["ts"]):
                f.write(json.dumps(entry) + "\n")

    # Generate summary
    summary = {
        "total_entries": len(all_entries),
        "phases": {},
        "items": {},
        "llm_calls": 0,
        "checks": {"passed": 0, "failed": 0},
        "fixes": 0
    }

    for entry in all_entries:
        phase = entry.get("phase", "unknown")
        event = entry.get("event", "unknown")
        item_key = f"{entry.get('item_id', '')}_{entry.get('request_id', '')}"

        # Count by phase
        if phase not in summary["phases"]:
            summary["phases"][phase] = 0
        summary["phases"][phase] += 1

        # Count by item
        if item_key not in summary["items"]:
            summary["items"][item_key] = 0
        summary["items"][item_key] += 1

        # Count event types
        if event == "llm_call":
            summary["llm_calls"] += 1
        elif event == "check":
            if entry.get("data", {}).get("passed"):
                summary["checks"]["passed"] += 1
            else:
                summary["checks"]["failed"] += 1
        elif event == "fix":
            summary["fixes"] += 1

    with open(f"{debug_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
