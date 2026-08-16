from .mock import MockProvider

def get_provider(name: str = "mock", **kwargs):
    """Factory for LLM providers. Falls back to Mock if keys/packages missing."""
    name = (name or "mock").lower()
    # Extract model if provided; avoid duplicate kwarg
    model = kwargs.pop("model", None)

    if name in ("mock", "mock-base-stock", "mock-noisy", "mock-random") or name.startswith("mock"):
        m = model or (name if name.startswith("mock-") else "mock-base-stock")
        return MockProvider(model=m, **kwargs)

    if name in ("openai", "gpt", "gpt-4o", "gpt-4o-mini"):
        try:
            from .openai_provider import OpenAIProvider
            m = model or "gpt-4o-mini"
            return OpenAIProvider(model=m, **kwargs)
        except Exception:
            return MockProvider(model=model or "mock-base-stock", **kwargs)

    if name in ("anthropic", "claude"):
        try:
            from .anthropic_provider import AnthropicProvider
            m = model or "claude-3-5-haiku-20241022"
            return AnthropicProvider(model=m, **kwargs)
        except Exception:
            return MockProvider(model=model or "mock-base-stock", **kwargs)

    return MockProvider(model=model or "mock-base-stock", **kwargs)

__all__ = ["MockProvider", "get_provider"]
