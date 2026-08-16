# Research Protocol — AgentSC-Bench

## 1. Research Questions

**Primary RQ**  
How do different architectures of LLM-based autonomous agents affect efficiency, coordination, reliability and resilience in multi-echelon supply chains under uncertainty and disruption?

**Secondary RQs**  
- RQ1: Can LLM agents outperform classical inventory policies?  
- RQ2: Does multi-agent communication improve performance vs independent agents?  
- RQ3: Does historical memory improve adaptation to recurring states?  
- RQ4: Does centralised information sharing reduce the bullwhip effect?  
- RQ5: How does agent autonomy affect supply-chain performance?  
- RQ6: How reliable are LLM decisions when the same state is presented repeatedly?  
- RQ7: How do agents behave under demand shocks, lead-time uncertainty and supplier disruptions?  
- RQ8: Do guardrails and structured decision policies improve reliability without large performance loss?  
- RQ9: What is the relationship between local agent optimisation and global supply-chain cost?

## 2. Hypotheses (pre-registered)

- H1: Communicating agents achieve lower system-wide cost than independent agents.  
- H2: Memory-enabled agents recover faster from recurring disruptions.  
- H3: Centralised information reduces bullwhip amplification.  
- H4: Higher autonomy does not necessarily improve global performance.  
- H5: Structured decision protocols improve reliability vs unrestricted outputs.  
- H6: Agent decision variance can amplify across tiers (agent bullwhip).  
- H7: Guardrails reduce catastrophic decision failures.

Hypotheses are tested, not assumed true.

## 3. Experimental Design

- **Independent variables**: architecture flags (classical / LLM / communication / memory / tools / guardrails / info mode / autonomy level), scenario, model, temperature, seed.  
- **Dependent variables**: total cost, service level, classical bullwhip ratios, Agent Bullwhip Index, reliability score, TTR, resilience score, communication volume, tool-call statistics.  
- **Controls**: identical demand seeds, identical initial inventories, multi-seed evaluation, MockProvider for offline runs.  
- **Baselines**: base-stock, order-up-to, safety-stock, moving-average.

## 4. Metrics Definitions (selected)

**Classical bullwhip** (Lee et al. inspired):  
`Var(Order_tier_i) / Var(CustomerDemand)`

**Agent Bullwhip Index** (project-defined, literature-inspired by Long et al. 2026):  
Average across tiers of (std of order quantities across repeated identical runs) / (mean order + ε).  
*This is not claimed to be the exact definition used in any prior paper.*

**Reliability score** (project-defined):  
`1 - min(1, CV(order quantities under identical state))`

**Time-to-Recovery (TTR)**: first post-disruption period at which period cost returns within (1+θ) of pre-disruption mean and remains stable for a short window.

## 5. Scenarios

See `experiments/scenarios.py` and `configs/`. All scenarios use deterministic seeds.

## 6. Statistical Methodology

- Multiple seeds (default ≥ 3).  
- Report mean, median, std, 95 % CI.  
- Welch’s t-test / Mann–Whitney U where assumptions are checked.  
- Effect sizes.  
- Multiple-comparison correction when appropriate.  
- No single-run claims of superiority.

## 7. Reproducibility

YAML config + seed + model + temperature + architecture flags + timestamp + (optional) git hash. MockProvider guarantees offline reproduction.

## 8. Limitations

- Beer-Game abstraction deliberately simplifies real multi-echelon networks.  
- MockProvider is a heuristic, not a real LLM; real-provider results may differ.  
- Autonomy and reliability scores are operational definitions open to refinement.  
- Phase 1 does not yet include communication, memory or real LLM providers.
