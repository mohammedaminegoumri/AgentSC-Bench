"""
Mock LLM provider.

Produces deterministic, schema-valid decisions without any external API.
Essential for unit tests, CI, and offline reproduction of the full pipeline.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional
import numpy as np

from ..base import LLMProvider, LLMResponse
from ..schemas import OrderDecisionSchema


class MockProvider(LLMProvider):
    """
    Deterministic mock that mimics a structured inventory decision.

    Behaviour modes (config via model name or kwargs):
    - "mock-base-stock" : classic base-stock heuristic
    - "mock-noisy"      : adds controlled noise (for reliability experiments)
    - "mock-random"     : higher variance (stress test)
    """

    def __init__(
        self,
        model: str = "mock-base-stock",
        temperature: float = 0.0,
        max_tokens: int = 256,
        seed: Optional[int] = 42,
        noise_std: float = 0.5,
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.seed = seed
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        self._call_log: list = []

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[Dict] = None,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        self.call_count += 1

        # Parse a few numbers from the user prompt for a realistic heuristic
        inv = self._extract_float(user_prompt, ["inventory", "Inventory"], default=12.0)
        backlog = self._extract_float(user_prompt, ["backlog", "Backlog"], default=0.0)
        demand = self._extract_float(user_prompt, ["incoming_order", "demand", "Demand"], default=4.0)
        ip = self._extract_float(user_prompt, ["inventory_position", "IP"], default=inv - backlog)

        # Adaptive base-stock: target ≈ demand * (lead_time + 1) + small buffer
        lt = self._extract_float(user_prompt, ["lead_time", "Lead time"], default=2.0)
        target = demand * (lt + 1) + 4.0
        base_order = max(0.0, target - ip)

        if "noisy" in self.model or self.temperature > 0:
            noise = self.rng.normal(0, self.noise_std * (1 + self.temperature))
            order = max(0.0, base_order + noise)
        elif "random" in self.model:
            order = max(0.0, self.rng.uniform(0, max(8.0, demand * 3)))
        else:
            order = base_order

        # Round to integer for realism
        order = float(round(order))

        decision = OrderDecisionSchema(
            order_quantity=order,
            confidence=0.85 if "noisy" not in self.model else 0.6,
            reasoning_summary=f"Mock heuristic: target={target:.1f}, IP={ip:.1f}, order={order:.1f}",
            risk_level="low" if backlog < 2 else "medium",
            communication_request=bool(backlog > 3),
        )

        content = decision.model_dump_json()
        latency = (time.perf_counter() - t0) * 1000

        # Fake token counts
        in_tok = len(system_prompt.split()) + len(user_prompt.split())
        out_tok = len(content.split())
        self.total_input_tokens += in_tok
        self.total_output_tokens += out_tok

        resp = LLMResponse(
            content=content,
            raw=decision.model_dump(),
            usage={"input_tokens": in_tok, "output_tokens": out_tok},
            latency_ms=latency,
            model=self.model,
        )
        self._call_log.append(resp)
        return resp

    @staticmethod
    def _extract_float(text: str, keys: list, default: float) -> float:
        import re
        for k in keys:
            # Look for patterns like "inventory: 12.0" or "inventory = 12"
            m = re.search(rf"{k}\s*[:=]\s*([0-9]+\.?[0-9]*)", text, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
        return default
