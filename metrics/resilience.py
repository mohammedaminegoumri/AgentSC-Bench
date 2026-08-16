"""Disruption recovery and resilience metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np


def compute_resilience(
    history: List[Dict[str, Any]],
    disruption_start: int,
    recovery_threshold: float = 0.1,
    pre_window: int = 5,
) -> Dict[str, float]:
    """
    Time-to-Recovery (TTR) and a simple Supply-Chain Resilience Score.

    TTR = first period after disruption when period cost returns within
          (1 + threshold) of the pre-disruption mean cost, and stays there.

    Resilience Score = 1 / (1 + normalised_max_degradation * (TTR / horizon))
    Clearly documented project definition; not claimed as a universal standard.
    """
    if not history or disruption_start >= len(history):
        return {"ttr": float("nan"), "resilience_score": 0.0}

    costs = np.array([h.get("total_cost", 0.0) for h in history], dtype=float)
    pre = costs[max(0, disruption_start - pre_window) : disruption_start]
    if len(pre) == 0:
        pre_mean = float(np.mean(costs[: disruption_start + 1]))
    else:
        pre_mean = float(np.mean(pre))

    post = costs[disruption_start:]
    if len(post) == 0:
        return {"ttr": 0.0, "resilience_score": 1.0, "pre_mean_cost": pre_mean}

    # Max degradation
    max_deg = float(np.max(post) - pre_mean)
    max_deg = max(0.0, max_deg)

    # TTR
    target = pre_mean * (1.0 + recovery_threshold)
    ttr = len(post)  # worst case: never recovers
    for i, c in enumerate(post):
        if c <= target:
            # Check stability for a short window
            window = post[i : i + 3]
            if len(window) >= 2 and np.all(window <= target * 1.05):
                ttr = i
                break

    horizon = max(1, len(history))
    norm_deg = max_deg / (pre_mean + 1e-6)
    resilience = 1.0 / (1.0 + norm_deg * (ttr / horizon))

    return {
        "ttr": float(ttr),
        "resilience_score": float(resilience),
        "pre_mean_cost": pre_mean,
        "max_degradation": max_deg,
        "post_mean_cost": float(np.mean(post)),
    }
