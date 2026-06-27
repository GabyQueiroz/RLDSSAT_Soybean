from __future__ import annotations

import argparse
from itertools import product

import numpy as np
import pandas as pd

from .config import load_config, make_paths
from .data import build_year_weather, load_observed_soybean_yield, load_weather, planting_date_for_year
from .dssat_adapter import PyDSSATRunner, build_irrigation_schedule


def _metrics(df: pd.DataFrame) -> dict:
    err = df["yield_kg_ha"] - df["observed_yield_kg_ha"]
    return {
        "mae_kg_ha": float(err.abs().mean()),
        "rmse_kg_ha": float(np.sqrt((err**2).mean())),
        "mape_percent": float((err.abs() / df["observed_yield_kg_ha"]).mean() * 100),
        "bias_kg_ha": float(err.mean()),
        "yield_mean_kg_ha": float(df["yield_kg_ha"].mean()),
        "irrigation_mean_mm": float(df["irrigation_mm"].mean()),
        "reward_mean": float(df["reward"].mean()),
        "n": int(len(df)),
    }


def _candidate_grid():
    planting_offsets = [10, 25, 40, 55, 70]
    triggers = [35, 50, 65]
    amounts = [15, 25]
    max_irrigations = [0, 90, 150]
    for offset, trigger, amount, max_irrig in product(planting_offsets, triggers, amounts, max_irrigations):
        if max_irrig == 0:
            trigger, amount = 9999, 0
        yield {
            "policy": f"offset{offset}_trig{trigger}_amt{amount}_max{max_irrig}",
            "planting_offset_days": offset,
            "trigger_dryness": trigger,
            "amount_mm": amount,
            "max_irrigation_mm": max_irrig,
        }


def evaluate_candidates(cfg: dict, paths, split: str, candidates: list[dict]) -> pd.DataFrame:
    weather = load_weather(paths.project_dir, cfg)
    observed = load_observed_soybean_yield(paths.project_dir, cfg).rename(columns={"ano": "year"})
    obs = dict(zip(observed["year"].astype(int), observed["observed_yield_kg_ha"].astype(float)))
    years = build_year_weather(weather, cfg["data"][f"{split}_years"])
    runner = PyDSSATRunner(cfg, paths.project_dir)
    rows = []
    rng = np.random.default_rng(cfg["seed"] + 7000)
    for candidate in candidates:
        for yw in years:
            if yw.year not in obs:
                continue
            planting = planting_date_for_year(yw.year, cfg["agronomy"]["planting_window_start"], candidate["planting_offset_days"])
            schedule = build_irrigation_schedule(
                yw.daily,
                planting,
                cfg["agronomy"]["season_length_days"],
                candidate["trigger_dryness"],
                candidate["amount_mm"],
                candidate["max_irrigation_mm"],
                cfg["agronomy"]["irrigation_check_days"],
            )
            sim = runner.run(yw.daily, planting, schedule, rng)
            reward = sim.yield_kg_ha / cfg["reward"]["target_yield_kg_ha"] - cfg["reward"]["water_penalty_per_mm"] * sim.irrigation_mm
            rows.append(
                {
                    "split": split,
                    "year": yw.year,
                    "observed_yield_kg_ha": obs[yw.year],
                    "yield_kg_ha": sim.yield_kg_ha,
                    "yield_error_kg_ha": sim.yield_kg_ha - obs[yw.year],
                    "yield_abs_error_kg_ha": abs(sim.yield_kg_ha - obs[yw.year]),
                    "irrigation_mm": sim.irrigation_mm,
                    "rain_mm": sim.rain_mm,
                    "planting_date": planting.isoformat(),
                    "reward": reward,
                    **candidate,
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Search fixed planting and irrigation rules with real DSSAT.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--run-name", default="management_search")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["backend"] = "dssat"
    paths = make_paths(cfg, args.run_name)
    cfg["dssat"]["keep_success_runs"] = False
    cfg["dssat"]["keep_failed_runs"] = False

    candidates = list(_candidate_grid())
    valid = evaluate_candidates(cfg, paths, "valid", candidates)
    valid.to_csv(paths.tables_dir / "management_search_valid_rows.csv", index=False, encoding="utf-8-sig")
    summary_rows = []
    for policy, group in valid.groupby("policy"):
        summary_rows.append({"policy": policy, **_metrics(group)})
    valid_summary = pd.DataFrame(summary_rows).sort_values(["mae_kg_ha", "irrigation_mean_mm"])
    valid_summary.to_csv(paths.tables_dir / "management_search_valid_summary.csv", index=False, encoding="utf-8-sig")

    best_policy = valid_summary.iloc[0]["policy"]
    best_candidate = [candidate for candidate in candidates if candidate["policy"] == best_policy]
    test = evaluate_candidates(cfg, paths, "test", best_candidate)
    test.to_csv(paths.tables_dir / "management_search_test_rows.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"split": "test", "policy": best_policy, **_metrics(test)}]).to_csv(
        paths.tables_dir / "management_search_test_summary.csv", index=False, encoding="utf-8-sig"
    )
    print("Best validation policy:", best_policy)
    print(valid_summary.head(10).to_string(index=False))
    print(test.to_string(index=False))


if __name__ == "__main__":
    main()
