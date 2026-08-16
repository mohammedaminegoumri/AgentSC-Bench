"""Inventory and forecasting tools available to agents."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np


class ToolRegistry:
    """
    Callable tools that agents may invoke. Every call is logged.
    Tools never mutate the environment; they only read or compute.
    """

    def __init__(self):
        self.call_log: List[Dict[str, Any]] = []

    def reset_log(self) -> None:
        self.call_log.clear()

    def _log(self, name: str, args: Dict, result: Any) -> None:
        self.call_log.append({"tool": name, "args": args, "result": result})

    def get_inventory(self, observation: Dict[str, Any]) -> float:
        val = float(observation.get("inventory", 0.0))
        self._log("get_inventory", {}, val)
        return val

    def get_backlog(self, observation: Dict[str, Any]) -> float:
        val = float(observation.get("backlog", 0.0))
        self._log("get_backlog", {}, val)
        return val

    def get_pipeline_inventory(self, observation: Dict[str, Any]) -> float:
        pipe = observation.get("pipeline", [])
        val = float(sum(pipe) if pipe else 0.0)
        self._log("get_pipeline_inventory", {}, val)
        return val

    def get_demand_history(self, history: List[float], window: int = 10) -> List[float]:
        vals = list(history[-window:]) if history else []
        self._log("get_demand_history", {"window": window}, vals)
        return vals

    def forecast_demand(self, history: List[float], method: str = "ma") -> float:
        if not history:
            result = 4.0
        elif method == "ma":
            result = float(np.mean(history[-5:]))
        else:
            result = float(history[-1])
        self._log("forecast_demand", {"method": method, "n": len(history)}, result)
        return result

    def calculate_safety_stock(
        self,
        demand_std: float,
        lead_time: int,
        z: float = 1.65,
    ) -> float:
        result = float(z * demand_std * np.sqrt(max(lead_time, 1)))
        self._log("calculate_safety_stock", {"demand_std": demand_std, "lead_time": lead_time, "z": z}, result)
        return result

    def calculate_reorder_point(
        self,
        avg_demand: float,
        lead_time: int,
        safety_stock: float,
    ) -> float:
        result = float(avg_demand * lead_time + safety_stock)
        self._log("calculate_reorder_point", {"avg_demand": avg_demand, "lead_time": lead_time, "ss": safety_stock}, result)
        return result

    def get_stats(self) -> Dict[str, Any]:
        return {
            "n_tool_calls": len(self.call_log),
            "tools_used": list({c["tool"] for c in self.call_log}),
        }
