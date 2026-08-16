"""Structured output schemas for LLM decisions. Invalid outputs are rejected."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, ValidationError


class OrderDecisionSchema(BaseModel):
    """
    Mandatory structured decision format.
    The LLM is never allowed to return free-form natural language as the action.
    """
    order_quantity: float = Field(..., ge=0.0, description="Non-negative order quantity")
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    reasoning_summary: str = Field("", max_length=500)
    risk_level: str = Field("medium")
    communication_request: bool = Field(False)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)

    @field_validator("risk_level")
    @classmethod
    def risk_must_be_valid(cls, v: str) -> str:
        allowed = {"low", "medium", "high"}
        v = v.lower().strip()
        if v not in allowed:
            return "medium"
        return v

    @field_validator("order_quantity")
    @classmethod
    def quantity_finite(cls, v: float) -> float:
        if not (v == v) or v == float("inf"):  # NaN or inf
            return 0.0
        return max(0.0, float(v))


def validate_decision(raw: Dict[str, Any] | str) -> OrderDecisionSchema:
    """
    Validate and coerce raw LLM output into a typed schema.
    Raises ValidationError on unrecoverable failure (caller should retry/fallback).
    """
    if isinstance(raw, str):
        # Attempt to extract JSON if the model wrapped it
        import json
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = json.loads(match.group(0))
        else:
            raise ValidationError.from_exception_data(
                "OrderDecisionSchema",
                [{"type": "value_error", "loc": ("raw",), "msg": "No JSON object found", "input": raw}],
            )
    return OrderDecisionSchema.model_validate(raw)
