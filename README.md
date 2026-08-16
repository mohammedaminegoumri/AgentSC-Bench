# AgentSC-Bench

**An Open Benchmark for Reliable Agentic AI in Multi-Echelon Supply Chains**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-orange.svg)](tests/)


**Author:** [Mohammed Amine Goumri](https://www.linkedin.com/in/mohammedaminegoumri/) · [GitHub](https://github.com/mohammedaminegoumri)

> Research-grade, reproducible experimental framework for evaluating whether (and under which conditions) LLM-based autonomous agents can effectively coordinate decisions in multi-echelon supply chains.

This is **not** a chatbot demo. It is a modular simulation + experiment harness designed for master's theses, technical reports, research portfolios, and potentially publishable work.

---

## Research Motivation

Recent work has shown that LLM agents can negotiate inventory decisions and reduce the bullwhip effect (Jannelli et al., 2024), that strong average performance can mask severe reliability problems ("agent bullwhip") (Long et al., 2026), that structured prompts and memory retrieval improve adaptation (Yoshizato et al., 2026), and that autonomy itself is a measurable, position-dependent factor (Trumpler et al., 2026).

**Gap:** There is no open, interchangeable experimental framework that lets researchers systematically compare classical inventory policies against a family of LLM-agent architectures (independent, communicating, memory-augmented, tool-using, guardrailed, centralised-information) under identical seeds, demand processes and disruption regimes while measuring efficiency, reliability, classical and agent-induced bullwhip, disruption recovery and autonomy–performance trade-offs.

AgentSC-Bench fills that gap.

## Core Research Question

> How do different architectures of LLM-based autonomous agents affect efficiency, coordination, reliability and resilience in multi-echelon supply chains under uncertainty and disruption?

See `docs/research_protocol.md` for the full set of secondary RQs and pre-registered hypotheses.

## Features

### Phase 1 (core) — complete
- Configurable multi-echelon Beer-Game-style simulator
- Classical baselines: base-stock, order-up-to, safety-stock, moving-average
- Provider-agnostic LLM layer with **MockProvider** (full pipeline runs without API keys)
- Structured decision schema (Pydantic) — free-form LLM text is never accepted as an action
- Standardised scenarios (stable, shocks, seasonal, supplier disruption, lead-time, …)
- Metrics: total cost, service level, classical bullwhip, Agent Bullwhip Index, reliability score, TTR / resilience
- YAML-driven multi-seed experiments, CSV + JSON output
- Unit tests with seed-reproducibility guarantees

### Phase 2 — complete
- Multi-agent communication bus (none / neighbor / full / central)
- Lightweight local experience memory with similarity retrieval
- Tool registry (inventory, forecast, safety stock, reorder point)
- MemoryAgent, ToolUsingAgent, GuardrailedAgent
- Optional real providers (OpenAI, Anthropic) — graceful fallback to Mock
- Reliability experiment mode (repeated identical states)
- Evaluation + publication-quality plots (PNG/SVG)
- Streamlit dashboard

## Installation

```bash
git clone https://github.com/mohammedaminegoumri/AgentSC-Bench.git
cd AgentSC-Bench
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

No paid API keys are required for default experiments.

## Quick Start

```bash
# Classical baseline
python -m experiments.runner --config configs/baseline.yaml

# Mock LLM agents (offline)
python -m experiments.runner --config configs/mock_llm.yaml

# Communicating agents
python -m experiments.runner --config configs/communicating.yaml

# Memory-augmented
python -m experiments.runner --config configs/memory.yaml

# Guardrailed
python -m experiments.runner --config configs/guardrailed.yaml

# Tool-using
python -m experiments.runner --config configs/tools.yaml

# Disruption scenario
python -m experiments.runner --config configs/disruption.yaml

# Reliability experiment
python -m experiments.runner --config configs/reliability.yaml --reliability --n-repeats 15

# Evaluate & plot
python -m experiments.evaluate --results-dir results

# Dashboard
streamlit run dashboard/app.py

# Tests
pytest -q
```

## Architecture

```
environment/     ← sole authority that validates & executes actions
agents/          ← classical, LLM, memory, tools, guardrails, communication
llm/             ← provider abstraction (Mock / OpenAI / Anthropic) + schemas
memory/          ← local similarity-based experience store
tools/           ← inventory & forecasting tools
metrics/         ← cost, bullwhip, reliability, resilience
experiments/     ← scenarios, runner, evaluate
dashboard/       ← Streamlit app
configs/         ← YAML experiment definitions
```

**Agents propose; the environment validates and executes.** This separation is mandatory.

## Supported architectures (interchangeable via config)

| Flag | Description |
|------|-------------|
| `classical` | Base-stock / order-up-to / safety-stock / MA |
| `mock_llm` / `llm` / `independent` | Independent LLM agents |
| `communicating` | LLM agents + message bus |
| `memory` | Similarity-based experience memory |
| `tools` | Tool-using LLM agents |
| `guardrailed` | LLM + hard order constraints |

## Reproducibility

Every experiment records seed(s), full YAML config, model, temperature, architecture, timestamp and (optional) git hash. MockProvider guarantees offline reproduction.

## Research Integrity

- No fabricated results, citations, or statistical claims.
- If an experiment has not been run the documentation states “Not yet evaluated.”
- Results that differ from prior literature are reported as-is.

## Citation

```bibtex
@software{agentscbench2026,
  title = {AgentSC-Bench: An Open Benchmark for Reliable Agentic AI in Multi-Echelon Supply Chains},
  author = {Goumri, Mohammed Amine},
  year = {2026},
  url = {https://github.com/mohammedaminegoumri/AgentSC-Bench},
  note = {LinkedIn: https://www.linkedin.com/in/mohammedaminegoumri/}
}
```

## License

Apache License 2.0 — see `LICENSE`.

## Literature (selected)

Full BibTeX in `references.bib`:

- Jannelli et al. (2024) arXiv:2411.10184 — Agentic LLMs & consensus
- Long et al. (2026) arXiv:2605.17036 — Reliability & agent bullwhip
- Yoshizato et al. (2026) arXiv:2602.05524 — Structured prompts & memory
- Xu et al. (2024) — MAS + foundation models for autonomous SCs
- Trumpler et al. (2026) arXiv:2607.25405 — Autonomy assessment
- Sterman (1989) / Lee et al. (1997) — Beer Game & bullwhip

We clearly distinguish *literature reproduction*, *literature-inspired extension*, and *new experimental comparisons*.

---

*Optimised for scientific validity, clean engineering and meaningful experimental results — not for flashy demos.*
