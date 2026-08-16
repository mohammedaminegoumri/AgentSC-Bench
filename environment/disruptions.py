"""Disruption injection for resilience experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class DisruptionEvent:
    """A single disruption that activates at a given period."""
    start_period: int
    duration: int
    type: str  # "supplier", "lead_time", "capacity", "communication"
    target_echelon: Optional[str] = None
    magnitude: float = 0.0  # interpretation depends on type
    active: bool = False


class DisruptionManager:
    """
    Applies temporary changes to lead times, capacities, or other parameters.
    Designed so that recovery metrics (TTR, resilience score) can be computed.
    """

    def __init__(self, events: Optional[List[Dict[str, Any]]] = None):
        self.events: List[DisruptionEvent] = []
        if events:
            for e in events:
                self.events.append(
                    DisruptionEvent(
                        start_period=int(e["start_period"]),
                        duration=int(e.get("duration", 5)),
                        type=str(e["type"]),
                        target_echelon=e.get("target_echelon"),
                        magnitude=float(e.get("magnitude", 0.0)),
                    )
                )
        self._original: Dict[str, Dict[str, Any]] = {}

    def reset(self, seed: Optional[int] = None) -> None:
        for ev in self.events:
            ev.active = False
        self._original.clear()

    def apply(self, period: int, echelons: Dict[str, Any], rng: np.random.Generator) -> None:
        """Apply or deactivate disruptions for the current period."""
        for ev in self.events:
            if period == ev.start_period and not ev.active:
                self._activate(ev, echelons)
            elif ev.active and period >= ev.start_period + ev.duration:
                self._deactivate(ev, echelons)

    def _activate(self, ev: DisruptionEvent, echelons: Dict[str, Any]) -> None:
        ev.active = True
        target = ev.target_echelon
        if target is None or target not in echelons:
            # Default to manufacturer for supplier disruption
            target = list(echelons.keys())[-1]
        e = echelons[target]
        key = f"{target}:{ev.type}"
        if key not in self._original:
            self._original[key] = {
                "lead_time": e.lead_time,
                "capacity": e.capacity,
            }
        if ev.type == "lead_time":
            e.lead_time = max(1, int(e.lead_time + ev.magnitude))
            # Extend pipeline
            while len(e.pipeline) < e.lead_time:
                e.pipeline.append(0.0)
        elif ev.type == "capacity":
            if e.capacity is None:
                e.capacity = 20.0  # default baseline
            e.capacity = max(0.0, e.capacity * (1.0 - ev.magnitude))
        elif ev.type == "supplier":
            # Severe capacity reduction
            e.capacity = max(0.0, (e.capacity or 30.0) * (1.0 - max(0.5, ev.magnitude)))
        # communication failure is handled at the agent/communication layer

    def _deactivate(self, ev: DisruptionEvent, echelons: Dict[str, Any]) -> None:
        ev.active = False
        target = ev.target_echelon or list(echelons.keys())[-1]
        key = f"{target}:{ev.type}"
        if key in self._original:
            orig = self._original[key]
            e = echelons[target]
            e.lead_time = orig["lead_time"]
            e.capacity = orig["capacity"]
            # Trim pipeline if needed
            if len(e.pipeline) > e.lead_time:
                e.pipeline = e.pipeline[: e.lead_time]
