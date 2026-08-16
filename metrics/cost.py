"""Cost and service-level metrics."""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import pandas as pd


def compute_cost_metrics(history: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute aggregate cost and service metrics from episode history.

    history: list of state dicts (from SupplyChainState.to_dict())
    """
    if not history:
        return {}

    total_costs = [h.get("total_cost", 0.0) for h in history]
    service = [h.get("service_level", 1.0) for h in history]
    cum = history[-1].get("cumulative_cost", sum(total_costs))

    return {
        "total_cost": float(cum),
        "mean_period_cost": float(np.mean(total_costs)),
        "std_period_cost": float(np.std(total_costs)),
        "mean_service_level": float(np.mean(service)),
        "min_service_level": float(np.min(service)),
        "periods": len(history),
    }
