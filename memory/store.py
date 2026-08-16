"""Lightweight local experience memory with similarity retrieval.

Uses pure NumPy; no external vector DB required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Experience:
    state_vector: np.ndarray
    state_dict: Dict[str, Any]
    decision: Dict[str, Any]
    outcome: Dict[str, Any]
    period: int = 0
    cost: float = 0.0


class ExperienceMemory:
    """
    Store (state, decision, outcome) tuples and retrieve K nearest neighbours
    by Euclidean distance on a numeric feature vector.
    """

    FEATURE_KEYS = [
        "inventory", "backlog", "incoming_order", "inventory_position",
        "lead_time", "pipeline_sum",
    ]

    def __init__(self, capacity: int = 500, k: int = 5):
        self.capacity = capacity
        self.k = k
        self.experiences: List[Experience] = []

    def _vectorize(self, state: Dict[str, Any]) -> np.ndarray:
        pipeline = state.get("pipeline", [])
        vals = [
            float(state.get("inventory", 0.0)),
            float(state.get("backlog", 0.0)),
            float(state.get("incoming_order", 0.0)),
            float(state.get("inventory_position", state.get("inventory", 0.0) - state.get("backlog", 0.0))),
            float(state.get("lead_time", 2)),
            float(sum(pipeline) if pipeline else 0.0),
        ]
        return np.array(vals, dtype=np.float64)

    def add(
        self,
        state: Dict[str, Any],
        decision: Dict[str, Any],
        outcome: Dict[str, Any],
        period: int = 0,
        cost: float = 0.0,
    ) -> None:
        vec = self._vectorize(state)
        exp = Experience(
            state_vector=vec,
            state_dict=dict(state),
            decision=dict(decision),
            outcome=dict(outcome),
            period=period,
            cost=cost,
        )
        self.experiences.append(exp)
        if len(self.experiences) > self.capacity:
            self.experiences.pop(0)

    def retrieve(self, state: Dict[str, Any], k: Optional[int] = None) -> List[Experience]:
        if not self.experiences:
            return []
        k = k or self.k
        query = self._vectorize(state)
        dists = []
        for i, exp in enumerate(self.experiences):
            d = float(np.linalg.norm(query - exp.state_vector))
            dists.append((d, i))
        dists.sort(key=lambda x: x[0])
        return [self.experiences[i] for _, i in dists[:k]]

    def format_for_prompt(self, experiences: List[Experience]) -> str:
        if not experiences:
            return "No similar past experiences."
        lines = ["Similar past experiences (most similar first):"]
        for i, exp in enumerate(experiences, 1):
            order = exp.decision.get("order_quantity", "?")
            cost = exp.cost
            inv = exp.state_dict.get("inventory", "?")
            back = exp.state_dict.get("backlog", "?")
            lines.append(
                f"  {i}. inv={inv}, backlog={back} -> ordered {order}, period_cost≈{cost:.1f}"
            )
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.experiences)

    def clear(self) -> None:
        self.experiences.clear()
