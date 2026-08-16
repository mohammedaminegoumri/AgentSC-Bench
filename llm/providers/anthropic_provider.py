"""Anthropic provider (optional). Requires ANTHROPIC_API_KEY."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from ..base import LLMProvider, LLMResponse


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-3-5-haiku-20241022", temperature: float = 0.0, max_tokens: int = 512):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                key = os.environ.get("ANTHROPIC_API_KEY")
                if not key:
                    raise RuntimeError("ANTHROPIC_API_KEY not set")
                self._client = anthropic.Anthropic(api_key=key)
            except ImportError:
                raise RuntimeError("anthropic package not installed")
        return self._client

    def complete(self, system_prompt: str, user_prompt: str, response_format: Optional[Dict] = None) -> LLMResponse:
        t0 = time.perf_counter()
        self.call_count += 1
        try:
            client = self._get_client()
            resp = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            content = resp.content[0].text if resp.content else ""
            usage = {
                "input_tokens": getattr(resp.usage, "input_tokens", 0),
                "output_tokens": getattr(resp.usage, "output_tokens", 0),
            }
            self.total_input_tokens += usage["input_tokens"]
            self.total_output_tokens += usage["output_tokens"]
            return LLMResponse(
                content=content,
                raw=resp,
                usage=usage,
                latency_ms=(time.perf_counter() - t0) * 1000,
                model=self.model,
            )
        except Exception as e:
            return LLMResponse(content="", error=str(e), latency_ms=(time.perf_counter() - t0) * 1000, model=self.model)
