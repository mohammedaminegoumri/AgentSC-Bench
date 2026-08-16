"""Guardrails that constrain LLM decisions without removing the LLM."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import Agent, AgentDecision
from .llm_agent import LLMAgent
from llm.base import LLMProvider


class GuardrailPolicy:
    """Simple hard constraints applied after the LLM proposes an order."""

    def __init__(
        self,
        max_order: float = 50.0,
        min_order: float = 0.0,
        max_order_multiple_of_demand: float = 5.0,
        prevent_negative_inventory_target: bool = True,
    ):
        self.max_order = max_order
        self.min_order = min_order
        self.max_order_multiple_of_demand = max_order_multiple_of_demand
        self.prevent_negative_inventory_target = prevent_negative_inventory_target

    def apply(self, proposed: float, observation: Dict[str, Any]) -> tuple[float, list[str]]:
        reasons = []
        order = float(proposed)

        if order < self.min_order:
            order = self.min_order
            reasons.append("clipped_below_min")
        if order > self.max_order:
            order = self.max_order
            reasons.append("clipped_above_max")

        demand = float(observation.get("incoming_order", 4.0))
        if demand > 0 and order > demand * self.max_order_multiple_of_demand:
            order = demand * self.max_order_multiple_of_demand
            reasons.append("clipped_multiple_of_demand")

        return max(0.0, order), reasons


class GuardrailedAgent(LLMAgent):
    """
    LLM agent whose proposed order is post-processed by hard guardrails.
    Improves reliability / reduces catastrophic orders without removing autonomy entirely.
    """

    def __init__(
        self,
        name: str,
        echelon: str,
        provider: LLMProvider,
        config: Optional[Dict[str, Any]] = None,
        guardrail: Optional[GuardrailPolicy] = None,
        **kwargs,
    ):
        super().__init__(name=name, echelon=echelon, provider=provider, config=config, **kwargs)
        self.guardrail = guardrail or GuardrailPolicy(
            max_order=float(config.get("max_order", 50.0)) if config else 50.0,
        )

    def decide(self, observation: Dict[str, Any], messages: Optional[list] = None) -> AgentDecision:
        decision = super().decide(observation, messages)
        original = decision.order_quantity
        clipped, reasons = self.guardrail.apply(original, observation)
        decision.order_quantity = clipped
        decision.metadata["guardrail_applied"] = bool(reasons)
        decision.metadata["guardrail_reasons"] = reasons
        decision.metadata["original_order"] = original
        if reasons:
            decision.reasoning_summary += f" [guardrail: {', '.join(reasons)}]"
            decision.confidence = min(decision.confidence, 0.7)
        return decision
