from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .data import build_year_weather, load_weather, planting_date_for_year
from .dssat_adapter import MockDSSATRunner, PyDSSATRunner, build_irrigation_schedule


def evaluate_baselines(cfg, paths, split: str = "test") -> pd.DataFrame:
    daily = load_weather(paths.project_dir, cfg)
    years = build_year_weather(daily, cfg["data"][f"{split}_years"])
    runner = PyDSSATRunner(cfg, paths.project_dir) if cfg["backend"] == "dssat" else MockDSSATRunner(cfg)
    ag = cfg["agronomy"]
    rng = np.random.default_rng(cfg["seed"] + 30_000)
    policies = [
        {"policy": "rainfed_calendar_oct15", "offset": 30, "trigger": 9999, "amount": 0, "max_irrig": 0},
        {"policy": "fixed_calendar_oct15_moderate_irrig", "offset": 30, "trigger": 45, "amount": 18, "max_irrig": 120},
        {"policy": "early_sep25_moderate_irrig", "offset": 10, "trigger": 45, "amount": 18, "max_irrig": 120},
        {"policy": "late_nov10_moderate_irrig", "offset": 56, "trigger": 45, "amount": 18, "max_irrig": 120},
    ]
    rows = []
    for yw in years:
        for pol in policies:
            pdate = planting_date_for_year(yw.year, ag["planting_window_start"], pol["offset"])
            sched = build_irrigation_schedule(
                yw.daily,
                pdate,
                ag["season_length_days"],
                pol["trigger"],
                pol["amount"],
                pol["max_irrig"],
                ag["irrigation_check_days"],
            )
            sim = runner.run(yw.daily, pdate, sched, rng)
            reward = sim.yield_kg_ha / cfg["reward"]["target_yield_kg_ha"] - cfg["reward"]["water_penalty_per_mm"] * sim.irrigation_mm
            rows.append(
                {
                    "split": split,
                    "year": yw.year,
                    "policy": pol["policy"],
                    "planting_date": pdate.isoformat(),
                    "yield_kg_ha": sim.yield_kg_ha,
                    "irrigation_mm": sim.irrigation_mm,
                    "rain_mm": sim.rain_mm,
                    "reward": reward,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(paths.tables_dir / f"baseline_evaluation_{split}.csv", index=False, encoding="utf-8-sig")
    summary = (
        out.groupby("policy")
        .agg(
            reward_mean=("reward", "mean"),
            yield_mean_kg_ha=("yield_kg_ha", "mean"),
            irrigation_mean_mm=("irrigation_mm", "mean"),
            yield_std_kg_ha=("yield_kg_ha", "std"),
        )
        .reset_index()
    )
    summary.to_csv(paths.tables_dir / f"baseline_summary_{split}.csv", index=False, encoding="utf-8-sig")
    return out
