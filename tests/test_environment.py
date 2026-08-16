"""Unit tests for the supply-chain environment — seed reproducibility is critical."""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from environment.supply_chain import SupplyChainEnv
from environment.demand import DemandGenerator, DemandProcess
from agents.classical import BaseStockAgent, create_classical_agent


def test_reset_deterministic():
    env1 = SupplyChainEnv(seed=42)
    env2 = SupplyChainEnv(seed=42)
    s1 = env1.reset()
    s2 = env2.reset()
    assert s1.period == s2.period
    assert s1.echelons["retailer"].inventory == s2.echelons["retailer"].inventory


def test_step_with_classical_agents_reproducible():
    def run(seed):
        env = SupplyChainEnv(seed=seed, max_periods=10)
        agents = {
            name: create_classical_agent("base_stock", f"bs_{name}", name, {"target": 20.0})
            for name in env.echelon_names
        }
        state = env.reset()
        costs = []
        while not env.done:
            actions = {}
            for name, agent in agents.items():
                obs = env.get_local_observation(name)
                dec = agent.decide(obs)
                actions[name] = dec.order_quantity
            state = env.step(actions)
            costs.append(state.total_cost)
        return costs, state.cumulative_cost

    c1, tot1 = run(123)
    c2, tot2 = run(123)
    assert c1 == c2
    assert tot1 == tot2


def test_inventory_non_negative_after_steps():
    env = SupplyChainEnv(seed=7, max_periods=5)
    agents = {
        name: BaseStockAgent(f"a_{name}", name, {"target": 15.0})
        for name in env.echelon_names
    }
    env.reset()
    while not env.done:
        actions = {n: agents[n].decide(env.get_local_observation(n)).order_quantity for n in env.echelon_names}
        state = env.step(actions)
        for e in state.echelons.values():
            assert e.inventory >= -1e-6  # numerical tolerance


def test_demand_generator_reproducible():
    g1 = DemandGenerator(process=DemandProcess.STOCHASTIC, base_demand=4.0, noise_std=1.0)
    g2 = DemandGenerator(process=DemandProcess.STOCHASTIC, base_demand=4.0, noise_std=1.0)
    rng1 = np.random.default_rng(99)
    rng2 = np.random.default_rng(99)
    d1 = [g1.generate(t, rng1) for t in range(20)]
    d2 = [g2.generate(t, rng2) for t in range(20)]
    assert d1 == d2
