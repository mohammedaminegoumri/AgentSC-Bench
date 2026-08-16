"""
Bullwhip and Agent-Bullwhip metrics.

Classical bullwhip: Var(Order_i) / Var(Demand)
Agent Bullwhip Index (project-defined, literature-inspired):
    Amplification of decision variance across tiers and across repeated runs.
We clearly separate literature-inspired classical ratios from our project metric.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def compute_bullwhip(
    history: List[Dict[str, Any]],
    echelon_order: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Classical bullwhip ratios per echelon and a global summary.
    Demand is taken as customer_demand series.
    """
    if not history:
        return {}

    demands = np.array([h.get("customer_demand", 0.0) for h in history], dtype=float)
    demand_var = float(np.var(demands)) if len(demands) > 1 else 0.0

    # Collect order series per echelon
    # history entries contain echelons dict
    first = history[0]
    names = echelon_order or list(first.get("echelons", {}).keys())

    ratios = {}
    order_vars = {}
    for name in names:
        orders = []
        for h in history:
            e = h.get("echelons", {}).get(name, {})
            orders.append(e.get("outgoing_order", 0.0))
        orders = np.array(orders, dtype=float)
        ov = float(np.var(orders)) if len(orders) > 1 else 0.0
        order_vars[name] = ov
        if demand_var > 1e-9:
            ratios[f"bullwhip_{name}"] = ov / demand_var
        else:
            ratios[f"bullwhip_{name}"] = 0.0 if ov < 1e-9 else float("inf")

    # Global: mean of ratios, and manufacturer / retailer amplification
    valid = [v for v in ratios.values() if np.isfinite(v)]
    ratios["bullwhip_mean"] = float(np.mean(valid)) if valid else 0.0
    ratios["demand_variance"] = demand_var
    ratios.update({f"order_var_{k}": v for k, v in order_vars.items()})
    return ratios


class AgentBullwhipIndex:
    """
    Project-defined metric (inspired by Long et al. 2026 "agent bullwhip").

    For a set of repeated runs under identical initial conditions / demand seed:
        ABI = average over tiers of (std of order quantities across runs) / (mean order + eps)

    Higher ABI indicates greater decision instability that can propagate upstream.
    This is NOT claimed to be the exact definition used by any prior paper.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def compute(self, run_histories: List[List[Dict[str, Any]]], echelon_names: List[str]) -> Dict[str, float]:
        """
        run_histories: list of episode histories (one per independent run)
        """
        if not run_histories:
            return {"agent_bullwhip_index": 0.0}

        # For each echelon, collect the vector of mean order (or final-period order) across runs
        # We use the full time-series variance across runs
        tier_scores = []
        details = {}
        n_periods = min(len(h) for h in run_histories)

        for name in echelon_names:
            # Shape: (n_runs, n_periods)
            orders = np.zeros((len(run_histories), n_periods))
            for r, hist in enumerate(run_histories):
                for t in range(n_periods):
                    e = hist[t].get("echelons", {}).get(name, {})
                    orders[r, t] = e.get("outgoing_order", 0.0)

            # Across-run standard deviation, averaged over time
            std_across_runs = np.std(orders, axis=0)
            mean_order = np.mean(orders) + self.eps
            score = float(np.mean(std_across_runs) / mean_order)
            tier_scores.append(score)
            details[f"abi_{name}"] = score

        abi = float(np.mean(tier_scores)) if tier_scores else 0.0
        details["agent_bullwhip_index"] = abi
        return details
