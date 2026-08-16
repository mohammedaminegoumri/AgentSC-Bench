"""LLM-based supply-chain agent with structured output and retry logic."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import json

from .base import Agent, AgentDecision
from llm.base import LLMProvider
from llm.schemas import validate_decision, OrderDecisionSchema
from pydantic import ValidationError


DEFAULT_SYSTEM_PROMPT = """You are an autonomous inventory manager for one echelon of a multi-echelon supply chain.
Your goal is to minimise holding + backlog costs while maintaining service level.
You must respond with a single JSON object that strictly matches this schema:
{
  "order_quantity": <non-negative number>,
  "confidence": <0-1>,
  "reasoning_summary": "<short explanation>",
  "risk_level": "low" | "medium" | "high",
  "communication_request": <boolean>
}
Do not output any other text. Do not mutate state; only propose an order quantity.
"""


class LLMAgent(Agent):
    """
    Single-echelon LLM agent.
    Uses a provider-agnostic LLM and enforces structured output validation.
    Invalid outputs trigger retry; after max_retries a safe fallback is used.
    """

    def __init__(
        self,
        name: str,
        echelon: str,
        provider: LLMProvider,
        config: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        max_retries: int = 2,
    ):
        super().__init__(name, echelon, config)
        self.provider = provider
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_retries = max_retries

    def decide(self, observation: Dict[str, Any], messages: Optional[List[Dict]] = None) -> AgentDecision:
        user_prompt = self._build_user_prompt(observation, messages)
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = self.provider.complete(self.system_prompt, user_prompt)
                if resp.error:
                    raise RuntimeError(resp.error)
                schema = validate_decision(resp.content)
                decision = AgentDecision(
                    order_quantity=schema.order_quantity,
                    confidence=schema.confidence,
                    reasoning_summary=schema.reasoning_summary,
                    risk_level=schema.risk_level,
                    communication_request=schema.communication_request,
                    tool_calls=schema.tool_calls,
                    metadata={
                        "model": resp.model,
                        "latency_ms": resp.latency_ms,
                        "usage": resp.usage,
                        "attempt": attempt,
                    },
                )
                self.last_decision = decision
                return decision
            except (ValidationError, json.JSONDecodeError, RuntimeError, ValueError) as e:
                last_error = e
                # On retry we can add a corrective hint
                user_prompt += f"\n\nPrevious output was invalid ({e}). Return ONLY valid JSON matching the schema."

        # Fallback: safe zero-order with low confidence
        decision = AgentDecision(
            order_quantity=0.0,
            confidence=0.1,
            reasoning_summary=f"Fallback after validation failures: {last_error}",
            risk_level="high",
            metadata={"fallback": True, "error": str(last_error)},
        )
        self.last_decision = decision
        return decision

    def _build_user_prompt(self, observation: Dict[str, Any], messages: Optional[List[Dict]]) -> str:
        lines = [
            f"Current period: {observation.get('period', '?')}",
            f"Your echelon: {self.echelon}",
            f"Inventory: {observation.get('inventory', 0):.1f}",
            f"Backlog: {observation.get('backlog', 0):.1f}",
            f"Inventory position: {observation.get('inventory_position', 0):.1f}",
            f"Incoming order (demand): {observation.get('incoming_order', 0):.1f}",
            f"Pipeline: {observation.get('pipeline', [])}",
            f"Lead time: {observation.get('lead_time', 2)}",
            f"Holding cost: {observation.get('holding_cost', 0.5)}",
            f"Backlog cost: {observation.get('backlog_cost', 1.0)}",
        ]
        if messages:
            lines.append("Messages from other agents:")
            for m in messages:
                lines.append(f"  - {m}")
        lines.append("\nDecide the order quantity for this period.")
        return "\n".join(lines)
