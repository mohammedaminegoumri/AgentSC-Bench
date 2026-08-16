"""
Reliability metrics for repeated decisions under identical states.

Core idea (Long et al. inspired): even with the same observation, stochastic
agents may produce different orders. We quantify that variance.
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np


def compute_reliability(
    repeated_decisions: List[float],
    repeated_costs: List[float] | None = None,
) -> Dict[str, float]:
    """
    repeated_decisions: list of order quantities produced for the *same* state
    across independent calls / seeds / temperature samples.

    Reliability score (project definition):
        1 - normalised decision variance
    where normalised variance is min(1, std / (mean + eps)).
    Documented as a simple, interpretable starting point; alternatives are welcome.
    """
    if not repeated_decisions:
        return {"reliability_score": 0.0, "decision_std": 0.0, "decision_cv": 0.0}

    arr = np.array(repeated_decisions, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    cv = std / (abs(mean) + 1e-6)
    # Bounded reliability in [0, 1]
    reliability = float(max(0.0, 1.0 - min(1.0, cv)))

    out = {
        "reliability_score": reliability,
        "decision_mean": mean,
        "decision_std": std,
        "decision_cv": float(cv),
        "n_repeats": len(arr),
    }
    if repeated_costs is not None and len(repeated_costs) == len(arr):
        c = np.array(repeated_costs, dtype=float)
        out["cost_std"] = float(np.std(c))
        out["cost_cv"] = float(np.std(c) / (abs(np.mean(c)) + 1e-6))
    return out
