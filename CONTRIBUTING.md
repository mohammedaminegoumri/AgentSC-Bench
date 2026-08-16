# Contributing to AgentSC-Bench

Thank you for interest in improving this research benchmark.

## Principles

1. **Reproducibility first** — every change that affects numerical results must preserve seed-based determinism for classical and MockProvider agents.
2. **No fabricated results** — documentation never invents experimental numbers.
3. **Clear separation** — agents propose; environment validates and executes.
4. **Literature honesty** — distinguish reproduction, inspiration, and new experiments.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Pull requests

- Keep PRs focused (one architectural concern or metric family).
- Add or update unit tests.
- Update `docs/research_protocol.md` if experimental design changes.
- Do not commit API keys or large result files.

## Roadmap alignment

Phase 1 (current) is the MVP. Preferred contributions for Phase 2+:

- Real LLM provider wrappers (OpenAI, Anthropic, Gemini, Grok) that respect the structured schema.
- Multi-agent communication protocols.
- Local vector memory.
- Tool-use layer with logging.
- Guardrail policies.
- Streamlit dashboard pages.
- Statistical evaluation helpers (bootstrap CIs, effect sizes).
