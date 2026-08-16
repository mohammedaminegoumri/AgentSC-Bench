# Literature Review (Selected)

This document summarises the papers that ground AgentSC-Bench. Full BibTeX is in `references.bib`.

## Agentic LLMs and Consensus

Jannelli et al. (2024, arXiv:2411.10184) demonstrate that LLM agents can automate consensus-seeking in inventory management and reduce bullwhip effects when equipped with appropriate tools and negotiation frameworks. Code was open-sourced for a limited case study.

## Reliability and Agent Bullwhip

Long et al. (2026, arXiv:2605.17036) show that strong average performance of reasoning models can mask severe reliability risks. They introduce the notion of *agent bullwhip* — amplification of run-to-run decision instability — and propose GRPO post-training to improve reliability.

## Structured Prompts and Memory

Yoshizato et al. (2026, arXiv:2602.05524) evaluate fixed-ordering strategy prompts and introduce AIM-RM, a similarity-based memory retrieval agent that improves adaptation across supply-chain scenarios.

## Autonomous Supply Chains (MAS + Foundation Models)

Xu et al. (2024) provide conceptual and implementation-oriented treatments of multi-agent systems combined with foundation models for autonomous supply chains (IFAC-PapersOnLine; Computers in Industry / arXiv:2310.09435).

## Negotiation and Information

Kirshner et al. (2025/2026) study LLM agents in supply-chain contract bargaining under public, private, ambiguous and deceptive cost information, highlighting efficiency–fairness trade-offs.

## Autonomy Measurement

Trumpler et al. (2026, arXiv:2607.25405) propose the Agentic AI Autonomy Assessment (AAAA) framework based on delegation, consultation and collaboration dimensions and test it in a Beer-Game setting, finding positional effects of autonomy on cost.

## Classical Foundations

Sterman (1989) and Lee, Padmanabhan & Whang (1997) remain the canonical references for the Beer Distribution Game and the bullwhip effect.

## Positioning of AgentSC-Bench

AgentSC-Bench is **literature-inspired**, not a direct reproduction of any single paper. Its novelty is the open, modular experimental apparatus that makes the architectures and factors studied across the above works *interchangeable and statistically comparable* under controlled conditions.
