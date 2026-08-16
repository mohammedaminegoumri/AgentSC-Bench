"""Tests for memory, tools and communication (Phase 2)."""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from memory.store import ExperienceMemory
from tools.inventory import ToolRegistry
from agents.communication import CommunicationBus, Message, CommTopology
from agents.multi_agent import build_agent_fleet
from llm.providers.mock import MockProvider


def test_memory_retrieve():
    mem = ExperienceMemory(capacity=50, k=3)
    for i in range(10):
        state = {"inventory": float(i), "backlog": 0.0, "incoming_order": 4.0,
                 "inventory_position": float(i), "lead_time": 2, "pipeline": [0, 0]}
        mem.add(state, {"order_quantity": 5.0}, {"cost": 1.0}, period=i, cost=1.0)
    q = {"inventory": 5.0, "backlog": 0.0, "incoming_order": 4.0,
         "inventory_position": 5.0, "lead_time": 2, "pipeline": [0, 0]}
    hits = mem.retrieve(q, k=3)
    assert len(hits) == 3
    assert len(mem) == 10


def test_tools():
    reg = ToolRegistry()
    obs = {"inventory": 12.0, "backlog": 1.0, "pipeline": [2.0, 3.0]}
    assert reg.get_inventory(obs) == 12.0
    assert reg.get_pipeline_inventory(obs) == 5.0
    assert reg.calculate_safety_stock(1.0, 2) > 0
    assert reg.get_stats()["n_tool_calls"] >= 3


def test_communication_neighbor():
    bus = CommunicationBus(topology=CommTopology.NEIGHBOR,
                           echelon_order=["retailer", "wholesaler", "distributor", "manufacturer"])
    assert bus.allowed_receivers("retailer") == ["wholesaler"]
    assert set(bus.allowed_receivers("wholesaler")) == {"retailer", "distributor"}
    ok = bus.send(Message(sender="retailer", receiver="wholesaler",
                          content={"x": 1}, period=0))
    assert ok
    msgs = bus.receive("wholesaler")
    assert len(msgs) == 1
    assert bus.message_count == 1


def test_communication_none():
    bus = CommunicationBus(topology=CommTopology.NONE)
    assert bus.allowed_receivers("retailer") == []


def test_build_fleet():
    provider = MockProvider(seed=1)
    agents = build_agent_fleet("classical", ["retailer", "wholesaler"], provider, {"classical_policy": "base_stock"}, 1)
    assert "retailer" in agents
    agents2 = build_agent_fleet("memory", ["retailer"], provider, {}, 1)
    assert hasattr(agents2["retailer"], "experience_memory")
