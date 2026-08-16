"""Helpers for multi-agent setups with optional communication."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import Agent, AgentDecision
from .llm_agent import LLMAgent
from .memory_agent import MemoryAgent
from .tool_agent import ToolUsingAgent
from .guardrails import GuardrailedAgent
from .classical import create_classical_agent
from agents.communication import CommunicationBus, Message, CommTopology
from llm.base import LLMProvider
from llm.providers.mock import MockProvider


def build_agent_fleet(
    architecture: str,
    echelon_names: List[str],
    provider: Optional[LLMProvider] = None,
    config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Dict[str, Agent]:
    """
    Factory that builds a full set of agents for the given architecture flag.
    Supported architecture strings (Phase 1+2):
      classical, mock_llm, llm, memory, tools, guardrailed,
      independent, communicating
    """
    config = config or {}
    provider = provider or MockProvider(model=config.get("model", "mock-base-stock"), seed=seed)
    agents: Dict[str, Agent] = {}
    arch = architecture.lower()

    for name in echelon_names:
        if arch == "classical":
            policy = config.get("classical_policy", "base_stock")
            agents[name] = create_classical_agent(policy, f"{policy}_{name}", name, config.get("agent", {}))
        elif arch in ("mock_llm", "llm", "independent"):
            agents[name] = LLMAgent(f"llm_{name}", name, provider, config.get("agent", {}))
        elif arch == "memory":
            agents[name] = MemoryAgent(f"mem_{name}", name, provider, config.get("agent", {}))
        elif arch == "tools":
            agents[name] = ToolUsingAgent(f"tool_{name}", name, provider, config.get("agent", {}))
        elif arch == "guardrailed":
            agents[name] = GuardrailedAgent(f"guard_{name}", name, provider, config.get("agent", {}))
        elif arch == "communicating":
            # same as independent LLM; communication is handled by the bus in the runner
            agents[name] = LLMAgent(f"comm_{name}", name, provider, config.get("agent", {}))
        else:
            # fallback
            agents[name] = LLMAgent(f"llm_{name}", name, provider, config.get("agent", {}))
    return agents
