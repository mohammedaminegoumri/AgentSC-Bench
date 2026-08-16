"""Memory-augmented LLM agent."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .llm_agent import LLMAgent, DEFAULT_SYSTEM_PROMPT
from .base import AgentDecision
from llm.base import LLMProvider
from memory.store import ExperienceMemory


class MemoryAgent(LLMAgent):
    """
    LLM agent that retrieves similar past experiences and includes them
    in the decision context. Experiences are stored after each step.
    """

    def __init__(
        self,
        name: str,
        echelon: str,
        provider: LLMProvider,
        config: Optional[Dict[str, Any]] = None,
        memory_capacity: int = 300,
        memory_k: int = 5,
        **kwargs,
    ):
        super().__init__(name=name, echelon=echelon, provider=provider, config=config, **kwargs)
        self.experience_memory = ExperienceMemory(capacity=memory_capacity, k=memory_k)

    def decide(self, observation: Dict[str, Any], messages: Optional[List[Dict]] = None) -> AgentDecision:
        # Retrieve similar experiences
        similar = self.experience_memory.retrieve(observation)
        memory_text = self.experience_memory.format_for_prompt(similar)

        # Augment the user prompt via a temporary override of _build_user_prompt
        original_build = self._build_user_prompt

        def augmented_build(obs, msgs):
            base = original_build(obs, msgs)
            return base + "\n\n" + memory_text

        self._build_user_prompt = augmented_build  # type: ignore
        try:
            decision = super().decide(observation, messages)
        finally:
            self._build_user_prompt = original_build  # type: ignore

        decision.metadata["memory_hits"] = len(similar)
        return decision

    def update_memory(self, state: Dict[str, Any], decision: AgentDecision, outcome: Dict[str, Any]) -> None:
        super().update_memory(state, decision, outcome)
        self.experience_memory.add(
            state=state,
            decision=decision.to_dict(),
            outcome=outcome,
            period=state.get("period", 0),
            cost=float(outcome.get("cost", 0.0)),
        )

    def reset(self) -> None:
        super().reset()
        self.experience_memory.clear()
