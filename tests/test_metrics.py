import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metrics.reliability import compute_reliability
from metrics.bullwhip import compute_bullwhip, AgentBullwhipIndex


def test_reliability_perfect():
    r = compute_reliability([5.0, 5.0, 5.0, 5.0])
    assert r["reliability_score"] == 1.0
    assert r["decision_std"] == 0.0


def test_reliability_noisy():
    r = compute_reliability([1.0, 10.0, 2.0, 9.0])
    assert 0.0 <= r["reliability_score"] < 1.0


def test_agent_bullwhip_index_empty():
    abi = AgentBullwhipIndex()
    out = abi.compute([], ["retailer"])
    assert out["agent_bullwhip_index"] == 0.0
