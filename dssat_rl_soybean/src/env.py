from __future__ import annotations

from datetime import timedelta

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .data import YearWeather, planting_date_for_year
from .dssat_adapter import build_irrigation_schedule


class SoybeanDSSATEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: dict, years: list[YearWeather], runner, seed: int = 0):
        super().__init__()
        self.cfg = cfg
        self.years = years
        self.runner = runner
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(9,), dtype=np.float32)
        self.current: YearWeather | None = None
        self.last_info = {}

    def _decode_action(self, action: np.ndarray) -> dict:
        ag = self.cfg["agronomy"]
        start = planting_date_for_year(2001, ag["planting_window_start"], 0)
        end = planting_date_for_year(2001, ag["planting_window_end"], 0)
        window_days = (end - start).days
        planting_offset = int(round(((float(action[0]) + 1.0) / 2.0) * window_days))
        trigger_dryness = 10.0 + ((float(action[1]) + 1.0) / 2.0) * 80.0
        amount_mm = 2.0 + ((float(action[2]) + 1.0) / 2.0) * ag["max_single_irrigation_mm"]
        max_irrig = 20.0 + ((float(action[3]) + 1.0) / 2.0) * ag["max_season_irrigation_mm"]
        return {
            "planting_offset_days": planting_offset,
            "trigger_dryness": trigger_dryness,
            "amount_mm": amount_mm,
            "max_irrigation_mm": max_irrig,
        }

    def _obs(self, yw: YearWeather) -> np.ndarray:
        extra = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return np.concatenate([yw.features, extra]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        idx = int(self.rng.integers(0, len(self.years)))
        self.current = self.years[idx]
        return self._obs(self.current), {"year": self.current.year}

    def step(self, action):
        assert self.current is not None
        decoded = self._decode_action(np.asarray(action, dtype=np.float32))
        planting_date = planting_date_for_year(
            self.current.year,
            self.cfg["agronomy"]["planting_window_start"],
            decoded["planting_offset_days"],
        )
        schedule = build_irrigation_schedule(
            self.current.daily,
            planting_date,
            self.cfg["agronomy"]["season_length_days"],
            decoded["trigger_dryness"],
            decoded["amount_mm"],
            decoded["max_irrigation_mm"],
            self.cfg["agronomy"]["irrigation_check_days"],
        )
        try:
            sim = self.runner.run(self.current.daily, planting_date, schedule, self.rng)
            y_scaled = sim.yield_kg_ha / self.cfg["reward"]["target_yield_kg_ha"]
            water_penalty = self.cfg["reward"]["water_penalty_per_mm"] * sim.irrigation_mm
            reward = float(y_scaled - water_penalty)
            failed = False
        except Exception as exc:
            sim = None
            reward = float(self.cfg["reward"]["failed_run_penalty"])
            failed = True
            decoded["error"] = repr(exc)

        info = {
            "year": self.current.year,
            "failed": failed,
            **decoded,
        }
        if sim is not None:
            info.update(
                {
                    "yield_kg_ha": sim.yield_kg_ha,
                    "irrigation_mm": sim.irrigation_mm,
                    "rain_mm": sim.rain_mm,
                    "planting_date": sim.planting_date.isoformat(),
                    "n_irrigation_events": int(len(schedule)),
                    "harvest_window_end": (sim.planting_date + timedelta(days=self.cfg["agronomy"]["season_length_days"])).isoformat(),
                }
            )
        self.last_info = info
        return self._obs(self.current), reward, True, False, info
