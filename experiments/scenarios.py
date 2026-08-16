"""Standardised experimental scenarios with deterministic seeds."""

from __future__ import annotations

from typing import Any, Dict, List
from environment.demand import DemandProcess


SCENARIOS: Dict[str, Dict[str, Any]] = {
    "stable": {
        "description": "Constant demand, no disruptions",
        "demand": {"process": DemandProcess.CONSTANT, "base_demand": 4.0},
        "disruptions": [],
        "max_periods": 40,
    },
    "shock_up": {
        "description": "Sudden demand increase at period 20",
        "demand": {
            "process": DemandProcess.SHOCK_UP,
            "base_demand": 4.0,
            "shock_period": 20,
            "shock_magnitude": 6.0,
        },
        "disruptions": [],
        "max_periods": 50,
    },
    "shock_down": {
        "description": "Sudden demand decrease at period 20",
        "demand": {
            "process": DemandProcess.SHOCK_DOWN,
            "base_demand": 8.0,
            "shock_period": 20,
            "shock_magnitude": 4.0,
        },
        "disruptions": [],
        "max_periods": 50,
    },
    "stochastic": {
        "description": "Random demand around base",
        "demand": {"process": DemandProcess.STOCHASTIC, "base_demand": 4.0, "noise_std": 1.5},
        "disruptions": [],
        "max_periods": 50,
    },
    "seasonal": {
        "description": "Seasonal demand pattern",
        "demand": {
            "process": DemandProcess.SEASONAL,
            "base_demand": 5.0,
            "seasonal_amplitude": 2.5,
            "seasonal_period": 12,
            "noise_std": 0.5,
        },
        "disruptions": [],
        "max_periods": 60,
    },
    "supplier_disruption": {
        "description": "Manufacturer capacity shock",
        "demand": {"process": DemandProcess.CONSTANT, "base_demand": 4.0},
        "disruptions": [
            {
                "start_period": 15,
                "duration": 8,
                "type": "capacity",
                "target_echelon": "manufacturer",
                "magnitude": 0.7,
            }
        ],
        "max_periods": 50,
    },
    "lead_time_increase": {
        "description": "Lead-time increase at distributor",
        "demand": {"process": DemandProcess.CONSTANT, "base_demand": 4.0},
        "disruptions": [
            {
                "start_period": 18,
                "duration": 10,
                "type": "lead_time",
                "target_echelon": "distributor",
                "magnitude": 2,
            }
        ],
        "max_periods": 50,
    },
    "capacity_reduction": {
        "description": "Global capacity pressure",
        "demand": {"process": DemandProcess.STOCHASTIC, "base_demand": 5.0, "noise_std": 1.0},
        "disruptions": [
            {
                "start_period": 12,
                "duration": 15,
                "type": "capacity",
                "target_echelon": "manufacturer",
                "magnitude": 0.5,
            }
        ],
        "max_periods": 50,
    },
    "distribution_shift": {
        "description": "Demand mean shifts permanently",
        "demand": {
            "process": DemandProcess.DISTRIBUTION_SHIFT,
            "base_demand": 4.0,
            "shift_period": 25,
            "shift_new_mean": 7.0,
            "noise_std": 1.0,
        },
        "disruptions": [],
        "max_periods": 55,
    },
}


def get_scenario(name: str) -> Dict[str, Any]:
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario '{name}'. Available: {list(SCENARIOS)}")
    return SCENARIOS[name].copy()


def list_scenarios() -> List[str]:
    return list(SCENARIOS.keys())
