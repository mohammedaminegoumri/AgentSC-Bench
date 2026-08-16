"""Common agent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentDecision:
    """Structured decision returned by every agent."""
    order_quantity: float
    confidence: float = 1.0
    reasoning_summary: str = ""
    risk_level: str = "medium"  # low / medium / high
    communication_request: bool = False
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_quantity": self.order_quantity,
            "confidence": self.confidence,
            "reasoning_summary": self.reasoning_summary,
            "risk_level": self.risk_level,
            "communication_request": self.communication_request,
            "tool_calls": self.tool_calls,
            "metadata": self.metadata,
        }


class Agent(ABC):
    """
    Abstract base class for all supply-chain agents.

    The environment never trusts the agent to mutate state.
    The agent only proposes an AgentDecision; the environment validates
    and executes.
    """

    def __init__(self, name: str, echelon: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.echelon = echelon
        self.config = config or {}
        self.memory: List[Dict[str, Any]] = []
        self.last_decision: Optional[AgentDecision] = None

    @abstractmethod
    def decide(self, observation: Dict[str, Any], messages: Optional[List[Dict]] = None) -> AgentDecision:
        """Produce a structured order decision given an observation."""
        ...

    def observe(self, observation: Dict[str, Any]) -> None:
        """Optional pre-processing of observation (override if needed)."""
        pass

    def communicate(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optional outbound messages. Default: none."""
        return []

    def update_memory(self, state: Dict[str, Any], decision: AgentDecision, outcome: Dict[str, Any]) -> None:
        """Store experience for memory-enabled agents."""
        self.memory.append({
            "state": state,
            "decision": decision.to_dict(),
            "outcome": outcome,
        })

    def explain_decision(self) -> str:
        if self.last_decision is None:
            return "No decision yet."
        return self.last_decision.reasoning_summary or "No explanation provided."

    def reset(self) -> None:
        self.memory.clear()
        self.last_decision = None
