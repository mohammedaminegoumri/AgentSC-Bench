"""Abstract LLM provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMResponse:
    content: str
    raw: Any = None
    usage: Dict[str, int] = field(default_factory=dict)  # input_tokens, output_tokens
    latency_ms: float = 0.0
    model: str = ""
    error: Optional[str] = None


class LLMProvider(ABC):
    """
    Provider-agnostic interface.
    Concrete providers (OpenAI, Anthropic, Gemini, Grok, Mock) implement this.
    """

    def __init__(self, model: str = "mock", temperature: float = 0.0, max_tokens: int = 512):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[Dict] = None,
    ) -> LLMResponse:
        ...

    def reset_stats(self) -> None:
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
