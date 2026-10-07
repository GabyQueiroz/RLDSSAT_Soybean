from __future__ import annotations

from datetime import timedelta

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .data import YearWeather, planting_date_for_year
from .dssat_adapter import build_irrigation_schedule


class SoybeanDSSATEnv(gym.Env):
    """One-step contextual decision: observe the t0 context, choose sowing date and irrigation rule, simulate the season."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: dict,
        years: list[YearWeather],
        runner,
        seed: int = 0,
        obs_scaler: tuple[np.ndarray, np.ndarray] | None = None,
        obs_noise_std: float = 0.0,
    ):
        super().__init__()
        self.cfg = cfg
        self.years = years
        self.runner = runner
        self.rng = np.random.default_rng(seed)
        self.obs_scaler = obs_scaler
        self.obs_noise_std = float(obs_noise_std)
        self.legacy_obs = obs_scaler is None and cfg.get("decision", {}).get("context_mode") != "pre_season_enso"
        n_obs = len(years[0].features) + (3 if self.legacy_obs else 0)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(n_obs,), dtype=np.float32)
        self.current: YearWeather | None = None
        self.last_info = {}

    def _decode_action(self, action: np.ndarray) -> dict:
        ag = self.cfg["agronomy"]
        action = np.clip(action, -1.0, 1.0)
        start = planting_date_for_year(2001, ag["planting_window_start"], 0)
        end = planting_date_for_year(2001, ag["planting_window_end"], 0)
        window_days = (end - start).days
        trig_lo, trig_hi = ag.get("trigger_range_mm", [10.0, 90.0])
        cap_lo = float(ag.get("min_season_cap_mm", 20.0))
        planting_offset = int(round(((float(action[0]) + 1.0) / 2.0) * window_days))
        trigger_dryness = trig_lo + ((float(action[1]) + 1.0) / 2.0) * (trig_hi - trig_lo)
        amount_mm = ((float(action[2]) + 1.0) / 2.0) * ag["max_single_irrigation_mm"]
        max_irrig = cap_lo + ((float(action[3]) + 1.0) / 2.0) * ag["max_season_irrigation_mm"]
        if ag.get("round_actions", False):
            trigger_dryness = round(trigger_dryness)
            amount_mm = round(amount_mm * 2.0) / 2.0
            max_irrig = round(max_irrig / 5.0) * 5.0
        return {
            "planting_offset_days": planting_offset,
            "trigger_dryness": trigger_dryness,
            "amount_mm": amount_mm,
            "max_irrigation_mm": max_irrig,
        }

    def _obs(self, yw: YearWeather) -> np.ndarray:
        if self.legacy_obs:
            extra = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            return np.concatenate([yw.features, extra]).astype(np.float32)
        obs = yw.features.astype(np.float32)
        if self.obs_scaler is not None:
            mean, std = self.obs_scaler
            obs = (obs - mean) / std
        obs = np.nan_to_num(obs, nan=0.0)
        if self.obs_noise_std > 0:
            obs = obs + self.rng.normal(0.0, self.obs_noise_std, size=obs.shape).astype(np.float32)
        return np.clip(obs, -5.0, 5.0).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        idx = int(self.rng.integers(0, len(self.years)))
        self.current = self.years[idx]
        return self._obs(self.current), {"year": self.current.year}

    def step(self, action):
        assert self.current is not None
        ag = self.cfg["agronomy"]
        decoded = self._decode_action(np.asarray(action, dtype=np.float32))
        planting_date = planting_date_for_year(
            self.current.year,
            ag["planting_window_start"],
            decoded["planting_offset_days"],
        )
        schedule = build_irrigation_schedule(
            self.current.daily,
            planting_date,
            ag["season_length_days"],
            decoded["trigger_dryness"],
            decoded["amount_mm"],
            decoded["max_irrigation_mm"],
            ag["irrigation_check_days"],
            ag.get("min_irrigation_event_mm", 0.0),
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
                    "harvest_window_end": (sim.planting_date + timedelta(days=ag["season_length_days"])).isoformat(),
                }
            )
        self.last_info = info
        return self._obs(self.current), reward, True, False, info
