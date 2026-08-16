"""OpenAI provider (optional). Requires OPENAI_API_KEY and openai package."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from ..base import LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0, max_tokens: int = 512):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                key = os.environ.get("OPENAI_API_KEY")
                if not key:
                    raise RuntimeError("OPENAI_API_KEY not set")
                self._client = OpenAI(api_key=key)
            except ImportError:
                raise RuntimeError("openai package not installed. pip install openai")
        return self._client

    def complete(self, system_prompt: str, user_prompt: str, response_format: Optional[Dict] = None) -> LLMResponse:
        t0 = time.perf_counter()
        self.call_count += 1
        try:
            client = self._get_client()
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if response_format:
                kwargs["response_format"] = response_format
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            usage = {}
            if resp.usage:
                usage = {
                    "input_tokens": resp.usage.prompt_tokens or 0,
                    "output_tokens": resp.usage.completion_tokens or 0,
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
