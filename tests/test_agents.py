import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.classical import create_classical_agent, BaseStockAgent
from agents.llm_agent import LLMAgent
from llm.providers.mock import MockProvider
from llm.schemas import validate_decision, OrderDecisionSchema


def test_base_stock_decision():
    agent = BaseStockAgent("test", "retailer", {"target": 20.0})
    obs = {"inventory_position": 12.0, "inventory": 12.0, "backlog": 0.0}
    dec = agent.decide(obs)
    assert dec.order_quantity == 8.0
    assert dec.confidence == 1.0


def test_mock_provider_schema():
    provider = MockProvider(model="mock-base-stock", seed=42)
    resp = provider.complete("system", "inventory: 10.0 inventory_position: 10.0 incoming_order: 4.0")
    schema = validate_decision(resp.content)
    assert schema.order_quantity >= 0
    assert 0 <= schema.confidence <= 1


def test_llm_agent_with_mock():
    provider = MockProvider(seed=1)
    agent = LLMAgent("llm_r", "retailer", provider)
    obs = {
        "period": 0,
        "inventory": 12.0,
        "backlog": 0.0,
        "inventory_position": 12.0,
        "incoming_order": 4.0,
        "pipeline": [0.0, 0.0],
        "lead_time": 2,
        "holding_cost": 0.5,
        "backlog_cost": 1.0,
    }
    dec = agent.decide(obs)
    assert dec.order_quantity >= 0
    assert "fallback" not in dec.metadata


def test_schema_rejects_negative():
    try:
        OrderDecisionSchema(order_quantity=-5)
        assert False, "Should have raised"
    except Exception:
        pass
