"""Classical inventory control policies used as strong baselines."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional
import numpy as np

from .base import Agent, AgentDecision


class BaseStockAgent(Agent):
    """
    Base-stock (order-up-to) policy.
    Target inventory position S; order = max(0, S - inventory_position).
    """

    def __init__(self, name: str, echelon: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, echelon, config)
        self.target = float(self.config.get("target", 20.0))

    def decide(self, observation: Dict[str, Any], messages: Optional[list] = None) -> AgentDecision:
        ip = observation.get("inventory_position", observation.get("inventory", 0.0))
        order = max(0.0, self.target - ip)
        decision = AgentDecision(
            order_quantity=order,
            confidence=1.0,
            reasoning_summary=f"Base-stock: target={self.target}, IP={ip:.1f}, order={order:.1f}",
            risk_level="low",
        )
        self.last_decision = decision
        return decision


class OrderUpToAgent(Agent):
    """
    Classic order-up-to with demand forecast.
    S_t = forecast * (lead_time + 1) + safety buffer.
    """

    def __init__(self, name: str, echelon: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, echelon, config)
        self.lead_time = int(self.config.get("lead_time", 2))
        self.safety = float(self.config.get("safety", 4.0))
        self.history: deque = deque(maxlen=20)

    def decide(self, observation: Dict[str, Any], messages: Optional[list] = None) -> AgentDecision:
        demand = observation.get("incoming_order", 4.0)
        self.history.append(demand)
        forecast = float(np.mean(self.history)) if self.history else demand
        target = forecast * (self.lead_time + 1) + self.safety
        ip = observation.get("inventory_position", observation.get("inventory", 0.0))
        order = max(0.0, target - ip)
        decision = AgentDecision(
            order_quantity=order,
            confidence=0.95,
            reasoning_summary=f"Order-up-to: forecast={forecast:.1f}, target={target:.1f}, IP={ip:.1f}",
            risk_level="low",
        )
        self.last_decision = decision
        return decision


class SafetyStockAgent(Agent):
    """
    Safety-stock policy: reorder point = lead_time * avg_demand + z * sigma * sqrt(lead_time).
    """

    def __init__(self, name: str, echelon: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, echelon, config)
        self.lead_time = int(self.config.get("lead_time", 2))
        self.z = float(self.config.get("z", 1.65))  # ~95% service
        self.history: deque = deque(maxlen=20)

    def decide(self, observation: Dict[str, Any], messages: Optional[list] = None) -> AgentDecision:
        demand = observation.get("incoming_order", 4.0)
        self.history.append(demand)
        avg = float(np.mean(self.history)) if self.history else demand
        std = float(np.std(self.history)) if len(self.history) > 1 else 1.0
        reorder_point = avg * self.lead_time + self.z * std * np.sqrt(self.lead_time)
        ip = observation.get("inventory_position", observation.get("inventory", 0.0))
        # Order enough to reach reorder_point + one period demand
        order = max(0.0, reorder_point + avg - ip)
        decision = AgentDecision(
            order_quantity=order,
            confidence=0.9,
            reasoning_summary=f"Safety-stock: ROP={reorder_point:.1f}, IP={ip:.1f}, order={order:.1f}",
            risk_level="medium",
        )
        self.last_decision = decision
        return decision


class MovingAverageAgent(Agent):
    """Moving-average demand forecast + simple reorder."""

    def __init__(self, name: str, echelon: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, echelon, config)
        self.window = int(self.config.get("window", 5))
        self.multiplier = float(self.config.get("multiplier", 3.0))
        self.history: deque = deque(maxlen=self.window)

    def decide(self, observation: Dict[str, Any], messages: Optional[list] = None) -> AgentDecision:
        demand = observation.get("incoming_order", 4.0)
        self.history.append(demand)
        forecast = float(np.mean(self.history)) if self.history else demand
        order = max(0.0, forecast * self.multiplier)
        # Also react to backlog
        backlog = observation.get("backlog", 0.0)
        order += backlog * 0.5
        decision = AgentDecision(
            order_quantity=order,
            confidence=0.85,
            reasoning_summary=f"MA({self.window}): forecast={forecast:.1f}, order={order:.1f}",
            risk_level="medium",
        )
        self.last_decision = decision
        return decision


def create_classical_agent(
    policy: str,
    name: str,
    echelon: str,
    config: Optional[Dict[str, Any]] = None,
) -> Agent:
    """Factory for classical agents."""
    mapping = {
        "base_stock": BaseStockAgent,
        "order_up_to": OrderUpToAgent,
        "safety_stock": SafetyStockAgent,
        "moving_average": MovingAverageAgent,
    }
    cls = mapping.get(policy.lower())
    if cls is None:
        raise ValueError(f"Unknown classical policy: {policy}. Choose from {list(mapping)}")
    return cls(name=name, echelon=echelon, config=config)
