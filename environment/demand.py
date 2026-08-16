"""Demand generation processes for the supply-chain simulator."""

from __future__ import annotations

from enum import Enum
from typing import Optional
import numpy as np


class DemandProcess(str, Enum):
    CONSTANT = "constant"
    STOCHASTIC = "stochastic"
    SEASONAL = "seasonal"
    SHOCK_UP = "shock_up"
    SHOCK_DOWN = "shock_down"
    SPIKE = "spike"
    DISTRIBUTION_SHIFT = "distribution_shift"


class DemandGenerator:
    """
    Configurable demand process.

    All randomness is controlled by the provided RNG so experiments are
    fully reproducible given a seed.
    """

    def __init__(
        self,
        process: DemandProcess | str = DemandProcess.CONSTANT,
        base_demand: float = 4.0,
        noise_std: float = 1.0,
        seasonal_amplitude: float = 2.0,
        seasonal_period: int = 12,
        shock_period: int = 20,
        shock_magnitude: float = 8.0,
        spike_period: int = 15,
        spike_magnitude: float = 12.0,
        shift_period: int = 25,
        shift_new_mean: float = 7.0,
        min_demand: float = 0.0,
    ):
        self.process = DemandProcess(process) if isinstance(process, str) else process
        self.base_demand = float(base_demand)
        self.noise_std = float(noise_std)
        self.seasonal_amplitude = float(seasonal_amplitude)
        self.seasonal_period = int(seasonal_period)
        self.shock_period = int(shock_period)
        self.shock_magnitude = float(shock_magnitude)
        self.spike_period = int(spike_period)
        self.spike_magnitude = float(spike_magnitude)
        self.shift_period = int(shift_period)
        self.shift_new_mean = float(shift_new_mean)
        self.min_demand = float(min_demand)
        self._current_mean = self.base_demand

    def reset(self, seed: Optional[int] = None) -> None:
        self._current_mean = self.base_demand

    def generate(self, period: int, rng: np.random.Generator) -> float:
        if self.process == DemandProcess.CONSTANT:
            d = self.base_demand
        elif self.process == DemandProcess.STOCHASTIC:
            d = self.base_demand + rng.normal(0, self.noise_std)
        elif self.process == DemandProcess.SEASONAL:
            season = self.seasonal_amplitude * np.sin(2 * np.pi * period / self.seasonal_period)
            d = self.base_demand + season + rng.normal(0, self.noise_std * 0.5)
        elif self.process == DemandProcess.SHOCK_UP:
            d = self.base_demand + (self.shock_magnitude if period >= self.shock_period else 0.0)
            d += rng.normal(0, self.noise_std * 0.3)
        elif self.process == DemandProcess.SHOCK_DOWN:
            d = self.base_demand - (self.shock_magnitude if period >= self.shock_period else 0.0)
            d += rng.normal(0, self.noise_std * 0.3)
        elif self.process == DemandProcess.SPIKE:
            d = self.base_demand + (self.spike_magnitude if period == self.spike_period else 0.0)
            d += rng.normal(0, self.noise_std * 0.3)
        elif self.process == DemandProcess.DISTRIBUTION_SHIFT:
            if period >= self.shift_period:
                self._current_mean = self.shift_new_mean
            d = self._current_mean + rng.normal(0, self.noise_std)
        else:
            d = self.base_demand

        return float(max(self.min_demand, d))
