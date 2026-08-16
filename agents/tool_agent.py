"""Tool-using LLM agent."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from .llm_agent import LLMAgent, DEFAULT_SYSTEM_PROMPT
from .base import AgentDecision
from llm.base import LLMProvider
from tools.inventory import ToolRegistry


TOOL_SYSTEM_ADDON = """
You may use the following tool results that have already been computed for you
(see the "tool_results" section of the observation):
- inventory, backlog, pipeline inventory
- forecast_demand, safety_stock, reorder_point
Incorporate them into your order decision.
"""


class ToolUsingAgent(LLMAgent):
    """
    LLM agent that calls inventory/forecasting tools before deciding.
    Tool results are injected into the observation.
    """

    def __init__(
        self,
        name: str,
        echelon: str,
        provider: LLMProvider,
        config: Optional[Dict[str, Any]] = None,
        tools: Optional[ToolRegistry] = None,
        **kwargs,
    ):
        system = DEFAULT_SYSTEM_PROMPT + TOOL_SYSTEM_ADDON
        super().__init__(
            name=name,
            echelon=echelon,
            provider=provider,
            config=config,
            system_prompt=system,
            **kwargs,
        )
        self.tools = tools or ToolRegistry()
        self.demand_history: List[float] = []

    def decide(self, observation: Dict[str, Any], messages: Optional[List[Dict]] = None) -> AgentDecision:
        inv = self.tools.get_inventory(observation)
        back = self.tools.get_backlog(observation)
        pipe = self.tools.get_pipeline_inventory(observation)
        demand = float(observation.get("incoming_order", 4.0))
        self.demand_history.append(demand)
        forecast = self.tools.forecast_demand(self.demand_history)
        std = float(np.std(self.demand_history[-10:])) if len(self.demand_history) > 2 else 1.0
        lt = int(observation.get("lead_time", 2))
        ss = self.tools.calculate_safety_stock(std, lt)
        rop = self.tools.calculate_reorder_point(forecast, lt, ss)

        tool_results = {
            "inventory": inv,
            "backlog": back,
            "pipeline": pipe,
            "forecast_demand": forecast,
            "safety_stock": ss,
            "reorder_point": rop,
        }
        observation = {**observation, "tool_results": tool_results}

        decision = super().decide(observation, messages)
        decision.tool_calls = list(self.tools.call_log[-6:])
        decision.metadata["n_tool_calls"] = len(self.tools.call_log)
        return decision

    def reset(self) -> None:
        super().reset()
        self.tools.reset_log()
        self.demand_history.clear()
