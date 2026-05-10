"""
AARS — Swarm Memory Store
Persistent learning memory for the restoration pipeline.
Stores spectral diagnostics, agent decisions, QC verdicts,
and accumulated heuristics across multiple runs.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


MEMORY_DIR = Path("/data/memory")


class SwarmMemory:
    """Persistent memory that enables progressive self-improvement.
    
    Stores three categories of knowledge:
    1. Run History — what happened in each restoration attempt
    2. Heuristics  — learned rules (e.g., "files with cutoff < 12kHz need aggressive excitation")
    3. Failures    — what went wrong and how it was corrected
    """

    def __init__(self, memory_dir: Path = MEMORY_DIR):
        self.memory_dir = memory_dir
        self.history_file = memory_dir / "run_history.json"
        self.heuristics_file = memory_dir / "heuristics.json"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._history: List[Dict] = self._load(self.history_file, default=[])
        self._heuristics: Dict = self._load(self.heuristics_file, default={
            "version": 1,
            "rules": [],
            "tool_reliability": {},
            "avg_improvement": {}
        })

    def _load(self, path: Path, default: Any) -> Any:
        if path.exists():
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return default
        return default

    def _save(self):
        with open(self.history_file, "w") as f:
            json.dump(self._history, f, indent=2, default=str)
        with open(self.heuristics_file, "w") as f:
            json.dump(self._heuristics, f, indent=2, default=str)

    # ── Recording ──────────────────────────────────────────

    def record_run(self, entry: Dict):
        """Record a complete run with its spectral data, decisions, and QC results."""
        entry["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        entry["run_id"] = len(self._history) + 1
        self._history.append(entry)

        # Update tool reliability stats
        if "tool_results" in entry:
            for tool_name, result in entry["tool_results"].items():
                if tool_name not in self._heuristics["tool_reliability"]:
                    self._heuristics["tool_reliability"][tool_name] = {
                        "calls": 0, "successes": 0, "failures": 0
                    }
                stats = self._heuristics["tool_reliability"][tool_name]
                stats["calls"] += 1
                if result.get("success", False):
                    stats["successes"] += 1
                else:
                    stats["failures"] += 1

        # Update average improvement metrics
        if "qc_metrics" in entry and entry["qc_metrics"]:
            qc = entry["qc_metrics"]
            for metric_key in ["snr_delta_db", "bandwidth_delta_hz", "crest_delta_db"]:
                if metric_key in qc:
                    if metric_key not in self._heuristics["avg_improvement"]:
                        self._heuristics["avg_improvement"][metric_key] = {
                            "sum": 0.0, "count": 0
                        }
                    self._heuristics["avg_improvement"][metric_key]["sum"] += qc[metric_key]
                    self._heuristics["avg_improvement"][metric_key]["count"] += 1

        # Learn heuristics from failures
        if entry.get("qc_passed") is False and entry.get("failure_reason"):
            self._learn_from_failure(entry)

        self._save()

    def _learn_from_failure(self, entry: Dict):
        """Extract a heuristic rule from a failed run."""
        rule = {
            "learned_at": entry["timestamp"],
            "filename": entry.get("filename", "unknown"),
            "spectral_profile": {
                k: entry["spectral"].get(k)
                for k in ["cutoff_freq_hz", "noise_floor_db", "crest_factor_db"]
                if k in entry.get("spectral", {})
            },
            "failure_reason": entry["failure_reason"],
            "action": "RETRY_WITH_ADJUSTMENT"
        }
        self._heuristics["rules"].append(rule)

    # ── Retrieval ──────────────────────────────────────────

    def get_context_for_strategist(self, spectral_data: Dict) -> str:
        """Generate a context briefing for the Strategist agent.
        
        Includes:
        - Summary of past performance
        - Relevant heuristic rules
        - Tool reliability data
        """
        lines = ["=== SWARM MEMORY BRIEFING ==="]

        # Past performance summary
        total_runs = len(self._history)
        if total_runs > 0:
            successes = sum(1 for r in self._history if r.get("qc_passed"))
            lines.append(f"\nRun History: {total_runs} total, {successes} passed QC ({100*successes//max(total_runs,1)}% success rate)")
        else:
            lines.append("\nNo previous runs recorded. This is the first restoration.")

        # Average improvements achieved
        avg = self._heuristics.get("avg_improvement", {})
        if avg:
            lines.append("\nAverage Improvements Achieved:")
            for k, v in avg.items():
                if v["count"] > 0:
                    mean = v["sum"] / v["count"]
                    lines.append(f"  - {k}: {mean:+.2f} (over {v['count']} runs)")

        # Tool reliability
        reliability = self._heuristics.get("tool_reliability", {})
        if reliability:
            lines.append("\nTool Reliability:")
            for tool, stats in reliability.items():
                rate = 100 * stats["successes"] / max(stats["calls"], 1)
                lines.append(f"  - {tool}: {rate:.0f}% ({stats['calls']} calls)")

        # Relevant failure heuristics
        rules = self._heuristics.get("rules", [])
        if rules:
            # Find rules relevant to similar spectral profiles
            cutoff = spectral_data.get("cutoff_freq_hz", 20000)
            relevant = [r for r in rules if abs(r.get("spectral_profile", {}).get("cutoff_freq_hz", 0) - cutoff) < 3000]
            if relevant:
                lines.append(f"\n⚠ Relevant Past Failures ({len(relevant)} similar profiles):")
                for r in relevant[-3:]:  # Last 3 most recent
                    lines.append(f"  - [{r['learned_at']}] {r['failure_reason']}")

        lines.append("\n=== END BRIEFING ===")
        return "\n".join(lines)

    def get_total_runs(self) -> int:
        return len(self._history)

    def get_success_rate(self) -> float:
        if not self._history:
            return 0.0
        return sum(1 for r in self._history if r.get("qc_passed")) / len(self._history)
